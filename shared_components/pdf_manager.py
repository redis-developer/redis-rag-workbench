import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from redisvl.index import SearchIndex

from shared_components.logger import logger


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

    def __init__(self, redis_url: str, storage_dir: str = "pdf_storage"):
        logger.info(f"Initializing PDFManager with Redis URL: {redis_url}")
        self.redis_url = redis_url
        self.storage_dir = Path(storage_dir).resolve()  # Get absolute path
        self._ensure_storage_dir()
        self._ensure_search_index(redis_url)

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
        self.index = SearchIndex.from_dict({
            "index": {
                "name": "pdf_manager",
                "prefix": "pdf",
                "key_separator": ":",
                "storage_type": "json"
            },
            "fields": [
                {"name": "filename", "type": "tag"},
                {"name": "index_name", "type": "text"},
                {"name": "upload_date", "type": "text"},
                {"name": "file_size", "type": "numeric"},
                {"name": "chunking_technique", "type": "text"}
            ]
        }, redis_url=redis_url)
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

    def add_pdf(
        self, file, chunk_size: int, chunking_technique: str, total_chunks: int
    ) -> str:
        """Register a new PDF and store its file."""
        try:
            # Store the physical PDF file
            file_path = self._store_pdf_file(file)

            # Get file size in KB
            file_size = Path(file_path).stat().st_size // 1024

            # Create metadata
            pdf_metadata = PDFMetadata(
                filename=Path(file.name).name,
                index_name=self._generate_index_name(file.name),
                upload_date=datetime.now().isoformat(),
                chunk_size=chunk_size,
                chunking_technique=chunking_technique,
                total_chunks=total_chunks,
                file_size=file_size,
                file_path=str(file_path),
            )

            # Store metadata using RedisJSON
            key = self.index.load([pdf_metadata.__dict__], id_field="index_name")

            if not key:
                raise Exception("Failed to store metadata in Redis")

            return pdf_metadata.index_name

        except Exception as e:
            print(f"DEBUG: Error in add_pdf: {str(e)}")
            # Cleanup if needed
            if "file_path" in locals() and Path(file_path).exists():
                Path(file_path).unlink()
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
            data = self.index.client.json().get(f"pdf:{index_name}")
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
