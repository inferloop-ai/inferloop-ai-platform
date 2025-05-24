# scripts/document-management.sh
#!/bin/bash
# Document management utility script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOCUMENT_PROCESSOR_URL="http://localhost:8006"
ACCESS_TOKEN="${MCP_ACCESS_TOKEN:-secure-memory-token}"
UPLOAD_DIR="./data/documents"
PROCESSED_DIR="./data/processed"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Show usage
usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  upload <file> <agent_id> <session_id>     Upload and process document"
    echo "  search <query> <agent_id>                 Search processed documents"
    echo "  list <agent_id> [session_id]              List documents for agent"
    echo "  status <job_id>                           Check processing job status"
    echo "  delete <document_id>                      Delete document and chunks"
    echo "  stats                                     Show processing statistics"
    echo "  batch-upload <directory> <agent_id> <session_id>  Batch upload directory"
    echo "  cleanup-old                               Cleanup old processed files"
    echo ""
    echo "Examples:"
    echo "  $0 upload ./document.pdf research-agent-001 session-123"
    echo "  $0 search 'machine learning' research-agent-001"
    echo "  $0 batch-upload ./research-papers research-agent-001 project-alpha"
}

# Upload single document
upload_document() {
    local file_path="$1"
    local agent_id="$2"
    local session_id="$3"
    local tags="${4:-document,uploaded}"
    local chunk_size="${5:-1000}"
    
    if [ ! -f "$file_path" ]; then
        log_error "File not found: $file_path"
        return 1
    fi
    
    log_step "Uploading document: $(basename "$file_path")"
    
    response=$(curl -s -X POST "$DOCUMENT_PROCESSOR_URL/process" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -F "files=@$file_path" \
        -F "agent_id=$agent_id" \
        -F "session_id=$session_id" \
        -F "tags=$tags" \
        -F "chunk_size=$chunk_size" \
        -F "enable_nlp_analysis=true" \
        -F "store_in_memory=true")
    
    if echo "$response" | grep -q "job_id"; then
        job_id=$(echo "$response" | jq -r '.job_id')
        log_info "Document uploaded successfully. Job ID: $job_id"
        
        # Wait for completion
        log_step "Waiting for processing to complete..."
        wait_for_completion "$job_id"
    else
        log_error "Upload failed: $response"
        return 1
    fi
}

# Wait for job completion
wait_for_completion() {
    local job_id="$1"
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        response=$(curl -s "$DOCUMENT_PROCESSOR_URL/job/$job_id" \
            -H "Authorization: Bearer $ACCESS_TOKEN")
        
        status=$(echo "$response" | jq -r '.status')
        progress=$(echo "$response" | jq -r '.progress_percentage')
        current_step=$(echo "$response" | jq -r '.current_step // "Processing..."')
        
        if [ "$status" = "completed" ]; then
            log_info "✅ Processing completed successfully!"
            
            # Show results
            chunks_created=$(echo "$response" | jq -r '.chunks_created')
            embeddings_generated=$(echo "$response" | jq -r '.embeddings_generated')
            log_info "📊 Results: $chunks_created chunks, $embeddings_generated embeddings"
            return 0
        elif [ "$status" = "failed" ]; then
            log_error "❌ Processing failed!"
            errors=$(echo "$response" | jq -r '.errors[]?' 2>/dev/null || echo "Unknown error")
            log_error "Errors: $errors"
            return 1
        else
            printf "\r⏳ Progress: %.1f%% - %s" "$progress" "$current_step"
            sleep 5
            ((attempt++))
        fi
    done
    
    log_warning "Processing timed out after 5 minutes"
    return 1
}

# Search documents
search_documents() {
    local query="$1"
    local agent_id="$2"
    local limit="${3:-10}"
    
    log_step "Searching documents for: '$query'"
    
    response=$(curl -s -X POST "$DOCUMENT_PROCESSOR_URL/search" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        -F "query=$query" \
        -F "agent_id=$agent_id" \
        -F "limit=$limit" \
        -F "similarity_threshold=0.7")
    
    if echo "$response" | grep -q "results"; then
        count=$(echo "$response" | jq -r '.count')
        log_info "Found $count results:"
        
        echo "$response" | jq -r '.results[] | "📄 \(.document_filename) (similarity: \(.similarity_score | tonumber | . * 100 | floor)%)\n   \(.content[:100])...\n"'
    else
        log_error "Search failed: $response"
        return 1
    fi
}

# List documents
list_documents() {
    local agent_id="$1"
    local session_id="$2"
    local limit="${3:-50}"
    
    log_step "Listing documents for agent: $agent_id"
    
    url="$DOCUMENT_PROCESSOR_URL/documents/$agent_id?limit=$limit"
    if [ -n "$session_id" ]; then
        url="$url&session_id=$session_id"
    fi
    
    response=$(curl -s "$url" -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "documents"; then
        count=$(echo "$response" | jq -r '.count')
        log_info "Found $count documents:"
        
        echo "$response" | jq -r '.documents[] | "📄 \(.filename) (\(.chunk_count) chunks) - \(.created_at[:19])"'
    else
        log_error "Failed to list documents: $response"
        return 1
    fi
}

# Check job status
check_status() {
    local job_id="$1"
    
    log_step "Checking status for job: $job_id"
    
    response=$(curl -s "$DOCUMENT_PROCESSOR_URL/job/$job_id" \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "status"; then
        echo "$response" | jq '.'
    else
        log_error "Failed to get job status: $response"
        return 1
    fi
}

# Delete document
delete_document() {
    local document_id="$1"
    
    log_step "Deleting document: $document_id"
    
    response=$(curl -s -X DELETE "$DOCUMENT_PROCESSOR_URL/document/$document_id" \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "success"; then
        log_info "✅ Document deleted successfully"
        echo "$response" | jq -r '.message'
    else
        log_error "Failed to delete document: $response"
        return 1
    fi
}

# Show statistics
show_stats() {
    log_step "Fetching processing statistics..."
    
    response=$(curl -s "$DOCUMENT_PROCESSOR_URL/stats" \
        -H "Authorization: Bearer $ACCESS_TOKEN")
    
    if echo "$response" | grep -q "processing_stats"; then
        echo "$response" | jq '.processing_stats'
    else
        log_error "Failed to get statistics: $response"
        return 1
    fi
}

# Batch upload directory
batch_upload() {
    local directory="$1"
    local agent_id="$2"
    local session_id="$3"
    local tags="${4:-batch,uploaded}"
    
    if [ ! -d "$directory" ]; then
        log_error "Directory not found: $directory"
        return 1
    fi
    
    log_step "Batch uploading files from: $directory"
    
    # Supported file extensions
    extensions=("*.pdf" "*.docx" "*.doc" "*.txt" "*.csv" "*.xlsx" "*.md" "*.html")
    
    total_files=0
    successful_uploads=0
    failed_uploads=0
    
    for ext in "${extensions[@]}"; do
        for file in "$directory"/$ext; do
            if [ -f "$file" ]; then
                ((total_files++))
                
                log_info "Processing file $total_files: $(basename "$file")"
                
                if upload_document "$file" "$agent_id" "$session_id" "$tags"; then
                    ((successful_uploads++))
                else
                    ((failed_uploads++))
                fi
                
                # Small delay between uploads
                sleep 2
            fi
        done
    done
    
    log_info "📊 Batch upload completed:"
    log_info "  Total files: $total_files"
    log_info "  Successful: $successful_uploads"
    log_info "  Failed: $failed_uploads"
}

# Cleanup old processed files
cleanup_old() {
    local days="${1:-30}"
    
    log_step "Cleaning up files older than $days days..."
    
    # Cleanup upload directory
    if [ -d "$UPLOAD_DIR" ]; then
        find "$UPLOAD_DIR" -type f -mtime +$days -delete
        log_info "Cleaned up old files in $UPLOAD_DIR"
    fi
    
    # Cleanup processed directory
    if [ -d "$PROCESSED_DIR" ]; then
        find "$PROCESSED_DIR" -type f -mtime +$days -delete
        log_info "Cleaned up old files in $PROCESSED_DIR"
    fi
    
    # Cleanup temp directory
    if [ -d "./data/temp" ]; then
        find "./data/temp" -type f -mtime +1 -delete
        log_info "Cleaned up temp files"
    fi
}

# Main command dispatcher
main() {
    if [ $# -lt 1 ]; then
        usage
        exit 1
    fi
    
    command="$1"
    shift
    
    case "$command" in
        upload)
            if [ $# -lt 3 ]; then
                log_error "Usage: $0 upload <file> <agent_id> <session_id> [tags] [chunk_size]"
                exit 1
            fi
            upload_document "$@"
            ;;
        search)
            if [ $# -lt 2 ]; then
                log_error "Usage: $0 search <query> <agent_id> [limit]"
                exit 1
            fi
            search_documents "$@"
            ;;
        list)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 list <agent_id> [session_id] [limit]"
                exit 1
            fi
            list_documents "$@"
            ;;
        status)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 status <job_id>"
                exit 1
            fi
            check_status "$@"
            ;;
        delete)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 delete <document_id>"
                exit 1
            fi
            delete_document "$@"
            ;;
        stats)
            show_stats
            ;;
        batch-upload)
            if [ $# -lt 3 ]; then
                log_error "Usage: $0 batch-upload <directory> <agent_id> <session_id> [tags]"
                exit 1
            fi
            batch_upload "$@"
            ;;
        cleanup-old)
            cleanup_old "$@"
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
