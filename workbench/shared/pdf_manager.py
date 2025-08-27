import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from redisvl.index import SearchIndex
from langchain_redis import RedisVectorStore

from workbench.shared.logger import logger
from workbench.shared.pdf_utils import process_file


@dataclass
class PDFMetadata:
    """Metadata for a stored PDF document."""

    filename: str
    index_name: str
    upload_date: str
    chunk_size: int
    chunking_technique: str
    total_chunks: int
    file_size: int
    file_path: str


class PDFManager:
    index: SearchIndex
    
    # System indexes to exclude from cleanup
    SYSTEM_INDEXES = {'pdf_manager', 'chat_history', 'semantic_router'}

    def __init__(self, redis_url: str, storage_dir: str = "data/pdfs"):
        logger.info(f"Initializing PDFManager with Redis URL: {redis_url}")
        self.redis_url = redis_url
        self.key_prefix = "pdf:manager"
        self.storage_dir = Path(storage_dir).resolve()
        self._ensure_storage_dir()
        self._ensure_search_index(redis_url)

    @property
    def client(self):
        """Redis client accessor."""
        return self.index.client

    def _ensure_storage_dir(self):
        """Ensure the PDF storage directory exists."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            gitkeep_file = self.storage_dir / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()
            logger.info(f"Storage directory ready: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise

    def _ensure_search_index(self, redis_url: str):
        """Create the search index for PDF metadata if it doesn't exist."""
        self.index = SearchIndex.from_dict(
            {
                "index": {
                    "name": "pdf_manager",
                    "prefix": self.key_prefix,
                    "key_separator": ":",
                    "storage_type": "json",
                },
                "fields": [
                    {"name": "filename", "type": "tag"},
                    {"name": "index_name", "type": "text"},
                    {"name": "upload_date", "type": "text"},
                    {"name": "file_size", "type": "numeric"},
                    {"name": "chunking_technique", "type": "text"},
                ],
            },
            redis_url=redis_url,
        )
        try:
            if not self.index.exists():
                self.index.create()
                logger.info("Search index created successfully")
        except Exception as e:
            logger.error(f"Error setting up search index: {e}")
            raise

    def _store_pdf_file(self, file) -> str:
        """Store the PDF file and return its path."""
        try:
            # Use the original filename directly
            original_filename = Path(file.name).name
            file_path = (self.storage_dir / original_filename).resolve()

            logger.info(f"Storing PDF at: {file_path}")

            # Copy file based on whether it's a file object or NamedString
            with open(file_path, "wb") as dest:
                if hasattr(file, "read"):
                    # File-like object with read method
                    dest.write(file.read())
                else:
                    # Gradio's NamedString - copy from the name path
                    with open(file.name, "rb") as src:
                        dest.write(src.read())

            if not file_path.exists():
                raise Exception(f"File was not stored at {file_path}")

            logger.info(f"Successfully stored PDF file: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store PDF file: {e}")
            raise

    def process_pdf_complete(self, file, chunk_size: int, chunking_technique: str, embeddings) -> str:
        """Complete PDF processing: store file, create metadata, and build vector store."""
        try:
            # Store the physical PDF file
            file_path = self._store_pdf_file(file)
            
            # Process the file to get documents
            documents, _ = process_file(file, chunk_size, chunking_technique, embeddings)
            
            # Get file size in KB
            file_size = Path(file_path).stat().st_size // 1024
            
            # Generate index name
            index_name = self._generate_index_name(file.name)

            # Create metadata
            pdf_metadata = PDFMetadata(
                filename=Path(file.name).name,
                index_name=index_name,
                upload_date=datetime.now().isoformat(),
                chunk_size=chunk_size,
                chunking_technique=chunking_technique,
                total_chunks=len(documents),
                file_size=file_size,
                file_path=str(file_path),
            )

            # Store metadata using RedisJSON (only if it doesn't exist)
            existing_metadata = self.get_pdf_metadata(index_name)
            if not existing_metadata:
                key = self.index.load([pdf_metadata.__dict__], id_field="index_name")
                if not key:
                    raise Exception("Failed to store metadata in Redis")
                logger.info(f"Created new metadata for {pdf_metadata.filename}")
            else:
                logger.info(f"Metadata already exists for {pdf_metadata.filename}, skipping creation")

            # Create vector store only if it doesn't exist
            if not self._check_vector_store_exists(index_name):
                vector_store = RedisVectorStore.from_documents(
                    documents,
                    embeddings,
                    redis_url=self.redis_url,
                    index_name=index_name,
                    key_prefix=f"pdf:{index_name}",
                )
                logger.info(f"Created new vector store for {index_name}")
            else:
                logger.info(f"Vector store already exists for {index_name}, skipping creation")

            logger.info(f"Successfully processed PDF: {pdf_metadata.filename}")
            return index_name

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            # Cleanup if needed
            if "file_path" in locals() and Path(file_path).exists():
                Path(file_path).unlink()
            raise

    def load_pdf_complete(self, index_name: str, embeddings) -> RedisVectorStore:
        """Load a PDF, reprocessing if needed. Returns the vector store."""
        try:
            # Get the metadata
            metadata = self.get_pdf_metadata(index_name)
            if not metadata:
                # Try to reprocess from file
                return self._reprocess_from_file(index_name, embeddings)

            # Check if vector store exists
            if not self._check_vector_store_exists(index_name):
                logger.info(f"Vector store missing for {index_name}, reprocessing")
                return self._reprocess_from_file(index_name, embeddings)

            # Vector store exists, load it
            vector_store = RedisVectorStore(
                embeddings,
                redis_url=self.redis_url,
                index_name=index_name,
                key_prefix=f"pdf:{index_name}",
            )
            
            logger.info(f"Successfully loaded PDF: {metadata.filename}")
            return vector_store

        except Exception as e:
            logger.error(f"Failed to load PDF {index_name}: {e}")
            raise

    def _reprocess_from_file(self, index_name: str, embeddings) -> RedisVectorStore:
        """Reprocess a PDF from its file on disk using existing metadata."""
        try:
            # Get existing metadata to preserve index_name
            metadata = self.get_pdf_metadata(index_name)
            if not metadata or not os.path.exists(metadata.file_path):
                raise Exception(f"Cannot reprocess {index_name}: metadata or file missing")

            logger.info(f"Reprocessing {metadata.filename} from disk")
            
            # Create a simple file-like object for process_file (it uses PyPDFLoader(file.name))
            class SimpleFile:
                def __init__(self, path):
                    self.name = str(path)  # PyPDFLoader needs this to be the file path
                    
            file_obj = SimpleFile(metadata.file_path)
            
            # Process the file to get documents
            documents, _ = process_file(file_obj, metadata.chunk_size, metadata.chunking_technique, embeddings)
            
            # Update metadata with new chunk count (only if different)
            if metadata.total_chunks != len(documents):
                metadata.total_chunks = len(documents)
                metadata.upload_date = datetime.now().isoformat()
                
                # Update metadata in Redis
                self.index.load([metadata.__dict__], id_field="index_name")
                logger.info(f"Updated metadata for {metadata.filename} with new chunk count")
            else:
                logger.info(f"Metadata for {metadata.filename} already up to date")
            
            # Create vector store with existing index_name (only if needed)
            if not self._check_vector_store_exists(index_name):
                vector_store = RedisVectorStore.from_documents(
                    documents,
                    embeddings,
                    redis_url=self.redis_url,
                    index_name=index_name,
                    key_prefix=f"pdf:{index_name}",
                )
                logger.info(f"Created vector store during reprocessing for {index_name}")
            else:
                # Load existing vector store
                vector_store = RedisVectorStore(
                    embeddings,
                    redis_url=self.redis_url,
                    index_name=index_name,
                    key_prefix=f"pdf:{index_name}",
                )
                logger.info(f"Loaded existing vector store for {index_name}")
            
            logger.info(f"Successfully reprocessed PDF: {metadata.filename}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to reprocess {index_name}: {e}")
            raise

    def search_pdfs(self, query: str = "*") -> List[PDFMetadata]:
        """Search PDFs using the Redis search index."""
        try:
            # Ensure we have a valid query
            if not query or query.strip() == "":
                query = "*"

            results = self.index.search(query)

            logger.info(f"Found {len(results.docs)} results")
            pdfs = []
            for doc in results.docs:
                try:
                    data = json.loads(doc.json)
                    pdfs.append(PDFMetadata(**data))
                except Exception as e:
                    logger.error(f"Error processing document {doc.id}: {e}")
                    continue

            return pdfs

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_pdf_metadata(self, index_name: str) -> Optional[PDFMetadata]:
        """Retrieve metadata for a specific PDF."""
        try:
            data = self.client.json().get(f"{self.key_prefix}:{index_name}")
            if data:
                return PDFMetadata(**data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving PDF metadata: {e}")
            return None

    def get_pdf_file(self, index_name: str) -> Optional[str]:
        """Get the file path for a stored PDF."""
        metadata = self.get_pdf_metadata(index_name)
        if metadata and os.path.exists(metadata.file_path):
            return metadata.file_path
        return None

    def _generate_index_name(self, filename: str) -> str:
        """Generate a consistent index name for a PDF."""
        base_name = Path(filename).stem
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name.lower())
        name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"{clean_name}_{name_hash}"

    def remove_pdf(self, index_name: str) -> bool:
        """Remove a PDF and its metadata from the system."""
        try:
            metadata = self.get_pdf_metadata(index_name)
            if not metadata:
                return False
            
            # Remove Redis metadata
            deleted = self.client.json().delete(f"{self.key_prefix}:{index_name}")
            if not deleted:
                return False
                
            # Clean up associated resources
            self._cleanup_vector_store(index_name)
            self._remove_file_safely(metadata.file_path)
            
            logger.info(f"Successfully removed PDF: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing PDF {index_name}: {e}")
            return False

    def _check_vector_store_exists(self, index_name: str) -> bool:
        """Check if a vector store exists for the given index name."""
        try:
            self.client.ft(index_name).info()
            return True
        except Exception:
            return False

    def _cleanup_vector_store(self, index_name: str) -> bool:
        """Clean up the vector store index and its documents."""
        if not self._check_vector_store_exists(index_name):
            return True
            
        try:
            self.client.ft(index_name).dropindex(delete_documents=True)
            logger.info(f"Successfully cleaned up vector store: {index_name}")
            return True
        except Exception as e:
            logger.warning(f"Could not clean up vector store {index_name}: {e}")
            return False

    def _remove_file_safely(self, file_path: str) -> None:
        """Safely remove a file with error handling."""
        if not file_path or not os.path.exists(file_path):
            return
            
        try:
            os.unlink(file_path)
            logger.info(f"Removed file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove file {file_path}: {e}")



    def _get_orphaned_vector_stores(self, known_indexes: set) -> List[str]:
        """Find vector store indexes that don't have corresponding PDF metadata."""
        try:
            all_indexes = self.client.execute_command("FT._LIST")
            orphaned = []
            
            for index_name in all_indexes:
                index_name = index_name.decode('utf-8') if isinstance(index_name, bytes) else index_name
                
                # Skip system indexes and known PDF indexes
                if (index_name not in self.SYSTEM_INDEXES and 
                    index_name not in known_indexes and
                    self._looks_like_pdf_index(index_name)):
                    orphaned.append(index_name)
                    
            return orphaned
            
        except Exception as e:
            logger.debug(f"Could not list Redis indexes: {e}")
            return []

    def _looks_like_pdf_index(self, index_name: str) -> bool:
        """Check if index name matches our PDF naming pattern."""
        return '_' in index_name and len(index_name) > 8

    def reconcile_data(self) -> Tuple[int, int, int]:
        """
        File-first reconciliation: Ensure all files on disk have proper Redis entries.
        
        Returns:
            Tuple of (files_processed, entries_cleaned, orphaned_cleaned)
        """
        logger.info("Starting file-first reconciliation...")
        
        try:
            # Phase 1: Ensure all files have metadata and vector stores
            files_processed = self._ensure_files_have_entries()
            
            # Phase 2: Clean up invalid Redis entries
            entries_cleaned = self._cleanup_invalid_entries()
            
            # Phase 3: Clean up orphaned vector stores
            orphaned_cleaned = self._cleanup_orphaned_vector_stores()
            
            logger.info(f"Reconciliation complete. Files processed: {files_processed}, Entries cleaned: {entries_cleaned}, Orphaned cleaned: {orphaned_cleaned}")
            return files_processed, entries_cleaned, orphaned_cleaned
            
        except Exception as e:
            logger.error(f"Error during reconciliation: {e}")
            return 0, 0, 0

    def _ensure_files_have_entries(self) -> int:
        """Ensure all PDF files have corresponding Redis metadata."""
        if not self.storage_dir.exists():
            return 0
            
        files_processed = 0
        
        for pdf_file in self.storage_dir.glob("*.pdf"):
            if pdf_file.name == ".gitkeep":
                continue
                
            index_name = self._generate_index_name(pdf_file.name)
            metadata = self.get_pdf_metadata(index_name)
            
            if not metadata:
                logger.info(f"Creating placeholder metadata for {pdf_file.name}")
                self._create_placeholder_metadata(pdf_file, index_name)
                files_processed += 1
            elif not self._check_vector_store_exists(index_name):
                logger.info(f"File {pdf_file.name} missing vector store - will be recreated on load")
                files_processed += 1
                
        return files_processed

    def _create_placeholder_metadata(self, pdf_file: Path, index_name: str) -> None:
        """Create placeholder metadata for a file that needs reprocessing."""
        try:
            file_size = pdf_file.stat().st_size // 1024
            
            # Create basic metadata with default values
            pdf_metadata = PDFMetadata(
                filename=pdf_file.name,
                index_name=index_name,
                upload_date=datetime.now().isoformat(),
                chunk_size=int(os.environ.get("DEFAULT_CHUNK_SIZE", 500)),
                chunking_technique=os.environ.get("DEFAULT_CHUNKING_TECHNIQUE", "Recursive Character"),
                total_chunks=0,  # Will be updated when actually processed
                file_size=file_size,
                file_path=str(pdf_file.resolve()),
            )
            
            # Store metadata using RedisJSON
            self.index.load([pdf_metadata.__dict__], id_field="index_name")
            logger.info(f"Created placeholder metadata for {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to create placeholder metadata for {pdf_file.name}: {e}")

    def _cleanup_invalid_entries(self) -> int:
        """Remove Redis entries that don't have corresponding files."""
        entries_cleaned = 0
        redis_pdfs = self.search_pdfs("*")
        
        for pdf in redis_pdfs:
            if not pdf.file_path or not os.path.exists(pdf.file_path):
                logger.info(f"Removing Redis entry for missing file: {pdf.filename}")
                if self.remove_pdf(pdf.index_name):
                    entries_cleaned += 1
                    
        return entries_cleaned

    def _cleanup_orphaned_vector_stores(self) -> int:
        """Clean up vector stores without corresponding metadata."""
        orphaned_count = 0
        redis_pdfs = self.search_pdfs("*")
        known_indexes = {pdf.index_name for pdf in redis_pdfs}
        
        for orphaned_index in self._get_orphaned_vector_stores(known_indexes):
            try:
                logger.info(f"Removing orphaned vector store: {orphaned_index}")
                self._cleanup_vector_store(orphaned_index)
                orphaned_count += 1
            except Exception as e:
                logger.warning(f"Could not remove orphaned vector store {orphaned_index}: {e}")
                
        return orphaned_count
