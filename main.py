"""FastMCP Server for Readwise Reader + Highlights Integration"""

import os
import json
import traceback
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from readwise_client import ReadwiseClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("Readwise MCP Enhanced")

# Get configuration from environment
READWISE_TOKEN = os.getenv("READWISE_TOKEN")
MCP_API_KEY = os.getenv("MCP_API_KEY")

if not READWISE_TOKEN:
    raise ValueError("READWISE_TOKEN environment variable is required")

if not MCP_API_KEY:
    logger.warning("MCP_API_KEY not set - server will run without authentication")

# Initialize Readwise client
client = ReadwiseClient(READWISE_TOKEN)

# Response size limit (100KB)
MAX_RESPONSE_SIZE = 100 * 1024


def format_json_response(data: Dict[str, Any], max_size: int = MAX_RESPONSE_SIZE) -> str:
    """
    Format response data as JSON string with size limits.
    
    Args:
        data: Dictionary to serialize
        max_size: Maximum response size in bytes (default: 100KB)
    
    Returns:
        JSON string, truncated if necessary
    """
    try:
        json_str = json.dumps(data, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        
        # Truncate if too large
        if len(json_bytes) > max_size:
            # Try to truncate the data array if it exists
            if 'results' in data and isinstance(data['results'], list):
                # Calculate how many items we can fit
                base_data = {k: v for k, v in data.items() if k != 'results'}
                base_json = json.dumps(base_data, ensure_ascii=False).encode('utf-8')
                available_size = max_size - len(base_json) - 200  # Reserve space for JSON structure and truncation metadata
                
                if available_size > 0:
                    # Binary search for optimal truncation
                    items = data['results']
                    low, high = 0, len(items)
                    while low < high:
                        mid = (low + high + 1) // 2
                        test_data = {**base_data, 'results': items[:mid]}
                        test_json = json.dumps(test_data, ensure_ascii=False).encode('utf-8')
                        if len(test_json) <= available_size:
                            low = mid
                        else:
                            high = mid - 1
                    
                    if low < len(items):
                        data['results'] = items[:low]
                        data['truncated'] = True
                        data['total_count'] = len(items)
                        json_str = json.dumps(data, ensure_ascii=False)
            else:
                # No results array to truncate, return error message
                json_str = json.dumps({"error": "Response too large", "truncated": True, "message": "Response exceeds maximum size limit"})
        
        return json_str
    except Exception as e:
        logger.error(f"Error formatting JSON response: {e}")
        logger.error(traceback.format_exc())
        # Fallback to simple error response
        try:
            return json.dumps({"error": str(e), "message": "Failed to format response"})
        except:
            return '{"error": "Failed to format response"}'


# ==================== Custom Authentication ====================
# Note: FastMCP 2.0+ handles auth differently
# We'll implement API key validation in the app setup below


# ==================== READER TOOLS (6) ====================

@mcp.tool()
async def readwise_save_document(
    url: str,
    tags: Optional[List[str]] = None,
    location: Optional[str] = "later",
    category: Optional[str] = "article"
) -> str:
    """
    Save a document to Readwise Reader.

    Args:
        url: The URL of the document to save
        tags: Optional list of tags to apply
        location: Where to save (new, later, archive, feed)
        category: Document category (article, email, rss, highlight, note, pdf, epub, tweet, video)

    Returns:
        JSON string with save result
    """
    try:
        kwargs = {}
        if tags:
            kwargs["tags"] = tags
        if location:
            kwargs["location"] = location
        if category:
            kwargs["category"] = category

        result = await client.save_document(url, **kwargs)
        return format_json_response({
            "success": True,
            "message": "Document saved successfully",
            "result": result
        })
    except Exception as e:
        logger.error(f"Error saving document: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to save document"})


@mcp.tool()
async def readwise_list_documents(
    location: Optional[str] = None,
    category: Optional[str] = None,
    author: Optional[str] = None,
    site_name: Optional[str] = None,
    limit: Optional[int] = 20,
    fetch_all: bool = False,
    updated_after: Optional[str] = None,
    with_full_content: bool = False,
    content_max_length: Optional[int] = None,
    max_limit: Optional[int] = 1000
) -> str:
    """
    List documents from Readwise Reader with advanced filtering and rate-limited fetch support.

    Args:
        location: Filter by location (new, later, archive, feed)
        category: Filter by category (article, email, rss, etc.)
        author: Filter by author name (case-insensitive partial match)
        site_name: Filter by site name (case-insensitive partial match)
        limit: Maximum documents to return (default: 20). Used when fetch_all=False
        fetch_all: If True, fetches documents up to max_limit (default: 1000)
        updated_after: ISO 8601 timestamp - only documents updated after this time
                      Example: "2025-11-01T00:00:00Z"
                      Useful for incremental syncs (fetch only new/updated docs)
        with_full_content: Include full document content (warning: may be large)
        content_max_length: Limit content length per document
        max_limit: Maximum documents to fetch when fetch_all=True (default: 1000)

    Returns:
        JSON string with filtered document list

    Examples:
        - Get documents by author: fetch_all=True, author="sukhad anand", max_limit=500
        - Get all LinkedIn posts: fetch_all=True, site_name="linkedin.com"
        - Get recent articles: updated_after="2025-11-01T00:00:00Z", category="article"
        - Incremental sync: fetch_all=True, updated_after="2025-11-28T00:00:00Z"
    """
    try:
        # Parameter validation
        if max_limit is not None and max_limit <= 0:
            return format_json_response({"error": "max_limit must be a positive integer"})
        if limit is not None and limit <= 0:
            return format_json_response({"error": "limit must be a positive integer"})
        
        # Fetch documents from API
        effective_limit = None if fetch_all else limit
        documents = await client.list_documents(
            location=location,
            category=category,
            limit=effective_limit,
            updated_after=updated_after,
            max_limit=max_limit if fetch_all else None
        )

        # Apply client-side filtering
        if author:
            author_lower = author.lower()
            documents = [
                doc for doc in documents
                if doc.get("author") and author_lower in doc["author"].lower()
            ]

        if site_name:
            site_lower = site_name.lower()
            documents = [
                doc for doc in documents
                if doc.get("site_name") and site_lower in doc["site_name"].lower()
            ]

        # Process content if requested
        if not with_full_content:
            for doc in documents:
                doc.pop("content", None)
        elif content_max_length:
            for doc in documents:
                if "content" in doc and len(doc["content"]) > content_max_length:
                    doc["content"] = doc["content"][:content_max_length] + "..."

        # Build response
        filters_applied = []
        if location:
            filters_applied.append(f"location={location}")
        if category:
            filters_applied.append(f"category={category}")
        if author:
            filters_applied.append(f"author contains '{author}'")
        if site_name:
            filters_applied.append(f"site contains '{site_name}'")
        if updated_after:
            filters_applied.append(f"updated after {updated_after}")

        return format_json_response({
            "count": len(documents),
            "results": documents,
            "filters_applied": filters_applied,
            "fetch_mode": "all" if fetch_all else "paginated"
        })
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to list documents"})


@mcp.tool()
async def readwise_update_document(
    document_id: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    summary: Optional[str] = None,
    location: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> str:
    """
    Update document metadata in Readwise Reader.

    Args:
        document_id: The ID of the document to update
        title: New title
        author: New author
        summary: New summary
        location: New location (new, later, archive, feed)
        tags: New tags list

    Returns:
        JSON string with update result
    """
    try:
        updates = {}
        if title:
            updates["title"] = title
        if author:
            updates["author"] = author
        if summary:
            updates["summary"] = summary
        if location:
            updates["location"] = location
        if tags:
            updates["tags"] = tags

        result = await client.update_document(document_id, updates)
        return format_json_response({
            "success": True,
            "message": "Document updated successfully",
            "document_id": document_id,
            "result": result
        })
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to update document"})


@mcp.tool()
async def readwise_delete_document(document_id: str) -> str:
    """
    Delete a document from Readwise Reader.

    Args:
        document_id: The ID of the document to delete

    Returns:
        Success or error message
    """
    try:
        await client.delete_document(document_id)
        return format_json_response({
            "success": True,
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id
        })
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to delete document"})


@mcp.tool()
async def readwise_list_tags() -> str:
    """
    Get all tags from Readwise Reader.

    Returns:
        JSON string with list of tags
    """
    try:
        tags = await client.list_tags()
        return format_json_response({
            "count": len(tags),
            "results": tags,
            "type": "tags"
        })
    except Exception as e:
        logger.error(f"Error listing tags: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to list tags"})


# ==================== HIGHLIGHTS TOOLS (7) ====================

@mcp.tool()
async def readwise_list_highlights(
    book_id: Optional[int] = None,
    page_size: int = 100,
    page: int = 1,
    fetch_all: bool = False,
    highlighted_at__gt: Optional[str] = None,
    highlighted_at__lt: Optional[str] = None,
    max_limit: Optional[int] = 5000
) -> str:
    """
    List highlights from Readwise with advanced filtering and rate-limited fetch support.

    Args:
        book_id: Filter by specific book ID
        page_size: Number of highlights per page (max 1000, ignored if fetch_all=True)
        page: Page number (ignored if fetch_all=True)
        fetch_all: If True, fetches highlights up to max_limit (default: 5000)
        highlighted_at__gt: Filter highlights after this date (ISO 8601)
        highlighted_at__lt: Filter highlights before this date (ISO 8601)
        max_limit: Maximum highlights to fetch when fetch_all=True (default: 5000)

    Returns:
        JSON string with highlights

    Examples:
        - Get highlights: fetch_all=True, max_limit=1000
        - Get highlights from specific book: fetch_all=True, book_id=12345
        - Get highlights from last week: highlighted_at__gt="2025-11-01T00:00:00Z"
    """
    try:
        # Parameter validation
        if max_limit is not None and max_limit <= 0:
            return format_json_response({"error": "max_limit must be a positive integer"})
        if page_size <= 0 or page_size > 1000:
            return format_json_response({"error": "page_size must be between 1 and 1000"})
        if page <= 0:
            return format_json_response({"error": "page must be a positive integer"})
        
        filters = {}
        if highlighted_at__gt:
            filters["highlighted_at__gt"] = highlighted_at__gt
        if highlighted_at__lt:
            filters["highlighted_at__lt"] = highlighted_at__lt

        result = await client.list_highlights(
            page_size=page_size,
            page=page,
            book_id=book_id,
            fetch_all=fetch_all,
            max_limit=max_limit if fetch_all else None,
            **filters
        )

        # Optimize response - only return essential fields
        highlights = result.get("results", [])
        optimized = [
            {
                "id": h.get("id"),
                "text": h.get("text"),
                "note": h.get("note"),
                "book_id": h.get("book_id"),
                "highlighted_at": h.get("highlighted_at")
            }
            for h in highlights
        ]

        total_count = result.get("count", len(optimized))
        return format_json_response({
            "count": total_count,
            "results": optimized,
            "fetch_mode": "all pages" if fetch_all else f"page {page}",
            "book_id": book_id
        })
    except Exception as e:
        logger.error(f"Error listing highlights: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to list highlights"})


@mcp.tool()
async def readwise_get_daily_review() -> str:
    """
    Get daily review highlights (spaced repetition learning system).

    Returns:
        JSON string with daily review highlights
    """
    try:
        result = await client.get_daily_review()

        # Optimize response
        highlights = result.get("highlights", [])
        optimized = [
            {
                "id": h.get("id"),
                "text": h.get("text"),
                "title": h.get("title"),
                "author": h.get("author"),
                "note": h.get("note")
            }
            for h in highlights
        ]

        return format_json_response({
            "count": len(optimized),
            "results": optimized,
            "type": "daily_review"
        })
    except Exception as e:
        logger.error(f"Error getting daily review: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to get daily review"})


@mcp.tool()
async def readwise_search_highlights(
    query: str,
    page_size: int = 100,
    page: int = 1,
    fetch_all: bool = False,
    max_limit: Optional[int] = 5000
) -> str:
    """
    Search highlights by text query with rate-limited fetch support.

    Args:
        query: Search term (searches highlight text and notes)
        page_size: Number of results per page (ignored if fetch_all=True)
        page: Page number (ignored if fetch_all=True)
        fetch_all: If True, fetches matching highlights up to max_limit (default: 5000)
        max_limit: Maximum highlights to fetch when fetch_all=True (default: 5000)

    Returns:
        JSON string with matching highlights

    Examples:
        - Search highlights: query="machine learning", fetch_all=True, max_limit=1000
        - Search first page: query="python", page_size=50
    """
    try:
        # Parameter validation
        if not query or not query.strip():
            return format_json_response({"error": "query cannot be empty"})
        if max_limit is not None and max_limit <= 0:
            return format_json_response({"error": "max_limit must be a positive integer"})
        if page_size <= 0 or page_size > 1000:
            return format_json_response({"error": "page_size must be between 1 and 1000"})
        if page <= 0:
            return format_json_response({"error": "page must be a positive integer"})
        
        result = await client.search_highlights(
            query=query.strip(),
            page_size=page_size,
            page=page,
            fetch_all=fetch_all,
            max_limit=max_limit if fetch_all else None
        )

        # Optimize response
        highlights = result.get("results", [])
        optimized = [
            {
                "id": h.get("id"),
                "text": h.get("text"),
                "book_id": h.get("book_id"),
                "note": h.get("note"),
                "title": h.get("title")
            }
            for h in highlights
        ]

        total_count = result.get("count", len(optimized))
        return format_json_response({
            "count": total_count,
            "results": optimized,
            "query": query.strip(),
            "fetch_mode": "all matches" if fetch_all else f"page {page}"
        })
    except Exception as e:
        logger.error(f"Error searching highlights: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to search highlights"})


@mcp.tool()
async def readwise_list_books(
    category: Optional[str] = None,
    page_size: int = 100,
    page: int = 1,
    fetch_all: bool = False,
    last_highlight_at__gt: Optional[str] = None,
    max_limit: Optional[int] = 1000
) -> str:
    """
    List books with highlight metadata and rate-limited fetch support.

    Args:
        category: Filter by category (books, articles, tweets, podcasts)
        page_size: Number of books per page (ignored if fetch_all=True)
        page: Page number (ignored if fetch_all=True)
        fetch_all: If True, fetches books up to max_limit (default: 1000)
        last_highlight_at__gt: Filter books with highlights after this date
        max_limit: Maximum books to fetch when fetch_all=True (default: 1000)

    Returns:
        JSON string with books

    Examples:
        - Get books: fetch_all=True, max_limit=500
        - Get all articles: fetch_all=True, category="articles"
        - Get books with recent highlights: last_highlight_at__gt="2025-11-01T00:00:00Z"
    """
    try:
        # Parameter validation
        if max_limit is not None and max_limit <= 0:
            return format_json_response({"error": "max_limit must be a positive integer"})
        if page_size <= 0 or page_size > 1000:
            return format_json_response({"error": "page_size must be between 1 and 1000"})
        if page <= 0:
            return format_json_response({"error": "page must be a positive integer"})
        
        filters = {}
        if last_highlight_at__gt:
            filters["last_highlight_at__gt"] = last_highlight_at__gt

        result = await client.list_books(
            page_size=page_size,
            page=page,
            category=category,
            fetch_all=fetch_all,
            max_limit=max_limit if fetch_all else None,
            **filters
        )

        # Optimize response
        books = result.get("results", [])
        optimized = [
            {
                "id": b.get("id"),
                "title": b.get("title"),
                "author": b.get("author"),
                "category": b.get("category"),
                "num_highlights": b.get("num_highlights")
            }
            for b in books
        ]

        total_count = result.get("count", len(optimized))
        return format_json_response({
            "count": total_count,
            "results": optimized,
            "category": category,
            "fetch_mode": "all pages" if fetch_all else f"page {page}"
        })
    except Exception as e:
        logger.error(f"Error listing books: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to list books"})


@mcp.tool()
async def readwise_get_book_highlights(book_id: int, max_limit: Optional[int] = 5000) -> str:
    """
    Get highlights from a specific book (automatically fetches multiple pages up to limit).

    Args:
        book_id: The ID of the book to get highlights from
        max_limit: Maximum highlights to fetch (default: 5000)

    Returns:
        JSON string with book highlights

    Example:
        - Get highlights from book: book_id=123456, max_limit=1000
    """
    try:
        # Parameter validation
        if book_id <= 0:
            return format_json_response({"error": "book_id must be a positive integer"})
        if max_limit is not None and max_limit <= 0:
            return format_json_response({"error": "max_limit must be a positive integer"})
        
        # This automatically fetches pages up to max_limit
        result = await client.get_book_highlights(book_id, max_limit=max_limit)

        highlights = result.get("results", [])
        optimized = [
            {
                "id": h.get("id"),
                "text": h.get("text"),
                "note": h.get("note"),
                "location": h.get("location"),
                "highlighted_at": h.get("highlighted_at")
            }
            for h in highlights
        ]

        total_count = result.get("count", len(optimized))
        return format_json_response({
            "count": total_count,
            "results": optimized,
            "book_id": book_id,
            "fetch_mode": "all pages"
        })
    except Exception as e:
        logger.error(f"Error getting book highlights: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to get book highlights"})


@mcp.tool()
async def readwise_export_highlights(
    updated_after: Optional[str] = None,
    include_deleted: bool = False,
    max_results: Optional[int] = 5000
) -> str:
    """
    Bulk export highlights for analysis and backup with rate limiting.

    This tool automatically fetches multiple pages of highlights up to max_results.
    For large libraries, use updated_after for incremental syncs.

    Args:
        updated_after: Export only highlights updated after this date (ISO 8601 format)
                      Example: "2025-11-01T00:00:00Z"
                      Tip: Use this for incremental syncs after initial full export
        include_deleted: Include deleted highlights in export
        max_results: Maximum number of highlights to export (default: 5000)
                    Set higher for larger exports, but be aware of rate limits

    Returns:
        JSON string with exported highlights

    Examples:
        - Export recent highlights: max_results=1000
        - Incremental since Nov 1: updated_after="2025-11-01T00:00:00Z", max_results=10000
        - Last week's changes: updated_after="2025-11-28T00:00:00Z"

    Note: Large exports may take time due to rate limiting delays between API calls
    """
    try:
        # Parameter validation
        if max_results is not None and max_results <= 0:
            return format_json_response({"error": "max_results must be a positive integer"})
        
        # Export fetches pages up to max_results with rate limiting
        highlights = await client.export_highlights(
            updated_after=updated_after,
            include_deleted=include_deleted,
            max_limit=max_results
        )

        # Optimize response - include more useful fields
        optimized = [
            {
                "id": h.get("id"),
                "text": h.get("text"),
                "title": h.get("title"),
                "author": h.get("author"),
                "book_id": h.get("book_id"),
                "note": h.get("note"),
                "highlighted_at": h.get("highlighted_at"),
                "updated": h.get("updated")
            }
            for h in highlights
        ]

        return format_json_response({
            "count": len(optimized),
            "results": optimized,
            "updated_after": updated_after,
            "include_deleted": include_deleted,
            "max_results": max_results
        })
    except Exception as e:
        logger.error(f"Error exporting highlights: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to export highlights"})


@mcp.tool()
async def readwise_create_highlight(
    text: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    note: Optional[str] = None,
    category: str = "books",
    highlighted_at: Optional[str] = None
) -> str:
    """
    Manually create a highlight in Readwise.

    Args:
        text: The highlight text (required)
        title: Book/article title
        author: Author name
        note: Your note on the highlight
        category: Category (books, articles, tweets, podcasts)
        highlighted_at: When it was highlighted (ISO 8601)

    Returns:
        JSON string with creation result
    """
    try:
        highlight_data = {"text": text}
        if title:
            highlight_data["title"] = title
        if author:
            highlight_data["author"] = author
        if note:
            highlight_data["note"] = note
        if category:
            highlight_data["category"] = category
        if highlighted_at:
            highlight_data["highlighted_at"] = highlighted_at

        result = await client.create_highlight([highlight_data])
        return format_json_response({
            "success": True,
            "message": "Highlight created successfully",
            "result": result
        })
    except Exception as e:
        logger.error(f"Error creating highlight: {e}")
        logger.error(traceback.format_exc())
        return format_json_response({"error": str(e), "message": "Failed to create highlight"})


# ==================== Server Entry Point ====================

def create_app():
    """Create the ASGI app with authentication wrapper"""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route, Mount

    async def health_check(request):
        return JSONResponse({
            "status": "healthy",
            "service": "readwise-mcp-enhanced",
            "version": "1.0.0",
            "authentication": "enabled" if MCP_API_KEY else "disabled"
        })

    async def auth_middleware(request, call_next):
        # Skip auth for health check and OAuth discovery endpoints
        if request.url.path in ["/health", "/.well-known/oauth-protected-resource",
                                "/.well-known/oauth-authorization-server", "/register"]:
            return await call_next(request)

        # Check API key if configured
        if MCP_API_KEY:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Missing or invalid Authorization header"},
                    status_code=401
                )

            token = auth_header.replace("Bearer ", "")
            if token != MCP_API_KEY:
                return JSONResponse(
                    {"error": "Invalid API key"},
                    status_code=401
                )

        return await call_next(request)

    # Get the FastMCP ASGI app
    mcp_app = mcp.http_app()

    # Create wrapper app with auth and CORS
    # IMPORTANT: Pass the FastMCP app's lifespan to Starlette
    # FastMCP's http_app() expects to handle requests at its root
    # So we mount it at / and it will handle /mcp endpoint internally
    app = Starlette(
        routes=[
            Route("/health", health_check, methods=["GET", "HEAD"]),
            Mount("/", mcp_app)  # FastMCP handles /mcp internally
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["https://claude.ai", "https://claude.com", "https://*.anthropic.com"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
        lifespan=mcp_app.lifespan  # Fix: Pass FastMCP's lifespan manager
    )

    # Add auth middleware
    @app.middleware("http")
    async def add_auth(request, call_next):
        return await auth_middleware(request, call_next)

    return app


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Remote Readwise MCP server on {host}:{port}")
    logger.info(f"Authentication: {'Enabled' if MCP_API_KEY else 'Disabled (WARNING: Not secure for production)'}")

    # Create and run the app
    app = create_app()
    uvicorn.run(app, host=host, port=port)
