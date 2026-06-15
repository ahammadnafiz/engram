#!/usr/bin/env bash
# =============================================================================
# Engram Docker Setup Script
# Auto-detects free port and starts PostgreSQL with pgvector
# =============================================================================
#
# Usage:
#   ./scripts/docker-setup.sh          # Auto-detect port, start postgres
#   ./scripts/docker-setup.sh --port 5433    # Use specific port
#   ./scripts/docker-setup.sh --down         # Stop and cleanup
#   ./scripts/docker-setup.sh --reset        # Reset (delete all data)
#   ./scripts/docker-setup.sh --status       # Check status
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
DEFAULT_PORT=5432
MAX_PORT_ATTEMPTS=20
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Check if a port is available
is_port_available() {
    local port="$1"
    if command -v ss &> /dev/null; then
        ! ss -tuln | grep -q ":${port} "
    elif command -v netstat &> /dev/null; then
        ! netstat -tuln | grep -q ":${port} "
    elif command -v lsof &> /dev/null; then
        ! lsof -i ":${port}" &> /dev/null
    else
        # Fallback: try to bind to the port
        (echo >/dev/tcp/localhost/"${port}") 2>/dev/null && return 1 || return 0
    fi
}

# Find a free port starting from a given port
find_free_port() {
    local start_port=${1:-$DEFAULT_PORT}
    local port=$start_port
    local attempts=0

    while [ $attempts -lt $MAX_PORT_ATTEMPTS ]; do
        if is_port_available $port; then
            echo $port
            return 0
        fi
        port=$((port + 1))
        attempts=$((attempts + 1))
    done

    log_error "Could not find a free port after $MAX_PORT_ATTEMPTS attempts"
    return 1
}

# Check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        echo "  Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    log_success "Docker is running"
}

# Check if docker compose is available
check_docker_compose() {
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        log_error "Docker Compose is not installed."
        exit 1
    fi
    log_success "Docker Compose available ($DOCKER_COMPOSE)"
}

# Generate a secure random password
generate_password() {
    # Try multiple methods for generating secure random password
    if command -v openssl &> /dev/null; then
        openssl rand -hex 16
    elif [ -r /dev/urandom ]; then
        head -c 32 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 16
    elif command -v python3 &> /dev/null; then
        python3 -c "import secrets; print(secrets.token_hex(16))"
    else
        log_warn "No secure random source found - using timestamp-based password"
        echo "engram_$(date +%s)_$$"
    fi
}

# Read a key from .env without sourcing secrets into this shell.
get_env_value() {
    local key=$1
    if [ -f "$ENV_FILE" ]; then
        awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE"
    fi
}

env_has_key() {
    local key=$1
    [ -f "$ENV_FILE" ] && grep -q -E "^${key}=" "$ENV_FILE"
}

append_env_value() {
    local key=$1
    local value=$2

    if [ -s "$ENV_FILE" ] && [ "$(tail -c 1 "$ENV_FILE")" != "" ]; then
        printf '\n' >> "$ENV_FILE"
    fi
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
}

set_env_value() {
    local key=$1
    local value=$2
    local tmp
    tmp=$(mktemp "${ENV_FILE}.XXXXXX")

    if [ -f "$ENV_FILE" ]; then
        awk -v key="$key" -v value="$value" '
            BEGIN { replaced = 0 }
            $0 ~ "^" key "=" && replaced == 0 {
                print key "=" value
                replaced = 1
                next
            }
            { print }
            END {
                if (replaced == 0) {
                    print key "=" value
                }
            }
        ' "$ENV_FILE" > "$tmp"
    else
        printf '%s=%s\n' "$key" "$value" > "$tmp"
    fi

    mv "$tmp" "$ENV_FILE"
}

is_managed_database_url() {
    local url=$1
    local user=$2
    local port=$3
    local db=$4

    [[ "$url" =~ ^postgresql://${user}:.+@localhost:${port}/${db}$ ]]
}

# Create .env file for first-time setup.
create_env_file() {
    local port=$1
    local password=${2:-$(generate_password)}

    cat > "$ENV_FILE" << EOF
# Engram Docker Configuration
# Generated by docker-setup.sh on $(date)

# PostgreSQL Settings
POSTGRES_PORT=$port
POSTGRES_USER=engram
POSTGRES_PASSWORD=$password
POSTGRES_DB=engram

# Engram Settings
ENGRAM_DATABASE_URL=postgresql://engram:${password}@localhost:${port}/engram

# Optional: OpenAI API Key (uncomment and add your key)
# OPENAI_API_KEY=sk-your-key-here

# Optional: Embedding Provider (openai or sentence-transformers)
EMBEDDING_PROVIDER=openai

# Optional: pgAdmin Settings
PGADMIN_PORT=5050
PGADMIN_EMAIL=admin@engram.local
PGADMIN_PASSWORD=admin

# Logging
LOG_LEVEL=INFO
EOF

    log_success "Created .env file with port $port"
}

create_or_update_env_file() {
    local port=$1
    local force_port_update=${2:-false}

    if [ ! -f "$ENV_FILE" ]; then
        create_env_file "$port"
        return
    fi

    local changed=false
    local existing_port
    existing_port=$(get_env_value POSTGRES_PORT)
    local postgres_user
    postgres_user=$(get_env_value POSTGRES_USER)
    postgres_user=${postgres_user:-engram}
    local postgres_db
    postgres_db=$(get_env_value POSTGRES_DB)
    postgres_db=${postgres_db:-engram}
    local postgres_password
    postgres_password=$(get_env_value POSTGRES_PASSWORD)

    if env_has_key POSTGRES_PORT; then
        if [ "$force_port_update" = "true" ] && [ "$existing_port" != "$port" ]; then
            set_env_value POSTGRES_PORT "$port"
            changed=true
        else
            port=$existing_port
        fi
    else
        append_env_value POSTGRES_PORT "$port"
        changed=true
    fi

    if ! env_has_key POSTGRES_USER; then
        append_env_value POSTGRES_USER "$postgres_user"
        changed=true
    fi

    if ! env_has_key POSTGRES_PASSWORD; then
        postgres_password=$(generate_password)
        append_env_value POSTGRES_PASSWORD "$postgres_password"
        changed=true
    fi

    if ! env_has_key POSTGRES_DB; then
        append_env_value POSTGRES_DB "$postgres_db"
        changed=true
    fi

    local database_url
    database_url=$(get_env_value ENGRAM_DATABASE_URL)
    local managed_url
    managed_url="postgresql://${postgres_user}:${postgres_password}@localhost:${port}/${postgres_db}"

    if ! env_has_key ENGRAM_DATABASE_URL; then
        append_env_value ENGRAM_DATABASE_URL "$managed_url"
        changed=true
    elif [ "$force_port_update" = "true" ] \
        && [ -n "$existing_port" ] \
        && is_managed_database_url "$database_url" "$postgres_user" "$existing_port" "$postgres_db"; then
        set_env_value ENGRAM_DATABASE_URL "$managed_url"
        changed=true
    fi

    if [ "$changed" = "true" ]; then
        log_success "Updated .env with missing Docker settings"
    else
        log_success "Using existing .env configuration"
    fi
}

# Load password from .env file
get_postgres_password() {
    get_env_value POSTGRES_PASSWORD
}

get_postgres_user() {
    local user
    user=$(get_env_value POSTGRES_USER)
    echo "${user:-engram}"
}

get_postgres_db() {
    local db
    db=$(get_env_value POSTGRES_DB)
    echo "${db:-engram}"
}

is_engram_postgres_running() {
    docker ps --format '{{.Names}}' | grep -Fxq "engram-postgres"
}

# Run psql command with password
run_psql() {
    local password
    password=$(get_postgres_password)
    local user
    user=$(get_postgres_user)
    local db
    db=$(get_postgres_db)
    $DOCKER_COMPOSE exec -T -e PGPASSWORD="$password" postgres psql -U "$user" -d "$db" "$@"
}

# Run pg_isready with password
run_pg_isready() {
    local password
    password=$(get_postgres_password)
    local user
    user=$(get_postgres_user)
    local db
    db=$(get_postgres_db)
    $DOCKER_COMPOSE exec -T -e PGPASSWORD="$password" postgres pg_isready -U "$user" -d "$db" "$@"
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    local max_attempts=30
    local attempt=1

    log_info "Waiting for PostgreSQL to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if run_pg_isready &> /dev/null; then
            log_success "PostgreSQL is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    log_error "PostgreSQL failed to start after $max_attempts seconds"
    return 1
}

# Verify database schema
verify_schema() {
    log_info "Verifying database schema..."

    local tables_raw
    tables_raw=$(run_psql -t -c \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")

    local tables
    tables=$(echo "$tables_raw" | tr -d ' \n')

    # Validate that we got a numeric result
    if ! [[ "$tables" =~ ^[0-9]+$ ]]; then
        log_warn "Could not verify schema (got: '$tables_raw')"
        return 1
    fi

    if [ "$tables" -ge 4 ]; then
        log_success "Database schema verified ($tables tables)"
        return 0
    else
        log_warn "Schema may be incomplete (only $tables tables found)"
        return 1
    fi
}

# Print connection info
print_connection_info() {
    local port=$1
    local user
    user=$(get_postgres_user)
    local db
    db=$(get_postgres_db)

    echo ""
    log_header "🎉 Engram is Ready!"
    echo ""
    echo -e "  ${GREEN}PostgreSQL:${NC}"
    echo -e "    Host:     localhost"
    echo -e "    Port:     $port"
    echo -e "    Database: $db"
    echo -e "    User:     $user"
    echo ""
    echo -e "  ${GREEN}Connection URL:${NC}"
    echo -e "    postgresql://$user:***@localhost:$port/$db"
    echo ""
    echo -e "  ${GREEN}Environment Variable:${NC}"
    echo -e "    export ENGRAM_DATABASE_URL=\"\$(grep ENGRAM_DATABASE_URL .env | cut -d= -f2-)\""
    echo ""
    echo -e "  ${GREEN}Python Usage:${NC}"
    echo -e "    from engram import Engram"
    echo -e "    async with Engram() as engram:"
    echo -e "        memory = await engram.add(content='Hello', agent_id='my_agent')"
    echo ""
    echo -e "  ${YELLOW}Commands:${NC}"
    echo -e "    ./scripts/docker-setup.sh --status   # Check status"
    echo -e "    ./scripts/docker-setup.sh --down     # Stop containers"
    echo -e "    ./scripts/docker-setup.sh --reset    # Reset database"
    echo ""
}

# -----------------------------------------------------------------------------
# Command Handlers
# -----------------------------------------------------------------------------

cmd_up() {
    local requested_port=${1:-}
    local port=""
    local force_port_update=false

    log_header "🚀 Starting Engram"

    check_docker
    check_docker_compose

    cd "$PROJECT_DIR"

    # Reuse .env on repeat runs so secrets and ports stay stable.
    if [ -n "$requested_port" ]; then
        port=$requested_port
        force_port_update=true
        if ! is_port_available "$port" && ! is_engram_postgres_running; then
            log_error "Port $port is already in use"
            exit 1
        fi
        log_success "Using specified port: $port"
    else
        port=$(get_env_value POSTGRES_PORT)
    fi

    # Find a free port for first-time setup only.
    if [ -z "$port" ]; then
        log_info "Auto-detecting free port..."
        port=$(find_free_port $DEFAULT_PORT)
        if [ $? -ne 0 ]; then
            exit 1
        fi
        log_success "Found free port: $port"
    else
        if [ "$force_port_update" != "true" ] && ! is_port_available "$port" && ! is_engram_postgres_running; then
            log_error "Port $port is already in use"
            exit 1
        fi
        if [ "$force_port_update" != "true" ]; then
            log_success "Using existing .env port: $port"
        fi
    fi

    # Create .env once, then preserve user-managed secrets on later runs.
    create_or_update_env_file "$port" "$force_port_update"
    port=$(get_env_value POSTGRES_PORT)

    # Start containers
    log_info "Starting PostgreSQL container..."
    POSTGRES_PORT=$port $DOCKER_COMPOSE up -d postgres

    # Wait for PostgreSQL
    wait_for_postgres
    if [ $? -ne 0 ]; then
        log_error "Failed to start PostgreSQL"
        $DOCKER_COMPOSE logs postgres
        exit 1
    fi

    # Verify schema
    sleep 2  # Give init scripts time to run
    verify_schema

    # Print connection info
    print_connection_info "$port"
}

cmd_down() {
    log_header "🛑 Stopping Engram"

    check_docker
    check_docker_compose

    cd "$PROJECT_DIR"

    log_info "Stopping containers..."
    $DOCKER_COMPOSE down

    log_success "Containers stopped"
}

cmd_reset() {
    log_header "🔄 Resetting Engram"

    check_docker
    check_docker_compose

    cd "$PROJECT_DIR"

    log_warn "This will delete all data!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Stopping and removing containers..."
        $DOCKER_COMPOSE down -v

        log_info "Removing .env file..."
        rm -f "$ENV_FILE"

        log_success "Reset complete"
        log_info "Run './scripts/docker-setup.sh' to start fresh"
    else
        log_info "Reset cancelled"
    fi
}

cmd_status() {
    log_header "📊 Engram Status"

    check_docker
    check_docker_compose

    cd "$PROJECT_DIR"

    echo ""
    log_info "Container Status:"
    $DOCKER_COMPOSE ps

    echo ""

    if run_pg_isready &> /dev/null; then
        log_success "PostgreSQL is healthy"

        # Show database stats
        local memory_count
        memory_count=$(run_psql -t -c "SELECT COUNT(*) FROM agent_memory;" 2>/dev/null || echo "0")
        memory_count=$(echo "$memory_count" | tr -d ' \n')

        local agent_count
        agent_count=$(run_psql -t -c "SELECT COUNT(*) FROM agents;" 2>/dev/null || echo "0")
        agent_count=$(echo "$agent_count" | tr -d ' \n')

        echo ""
        log_info "Database Stats:"
        echo "    Agents:   $agent_count"
        echo "    Memories: $memory_count"
    else
        log_warn "PostgreSQL is not running"
    fi

    if [ -f "$ENV_FILE" ]; then
        echo ""
        log_info "Configuration (.env):"
        grep -E "^(POSTGRES_PORT|ENGRAM_DATABASE_URL)" "$ENV_FILE" | sed 's/^/    /'
    fi
}

cmd_logs() {
    check_docker
    check_docker_compose

    cd "$PROJECT_DIR"
    $DOCKER_COMPOSE logs -f postgres
}

cmd_shell() {
    check_docker
    check_docker_compose

    cd "$PROJECT_DIR"
    log_info "Connecting to PostgreSQL shell..."
    local password
    password=$(get_postgres_password)
    local user
    user=$(get_postgres_user)
    local db
    db=$(get_postgres_db)
    $DOCKER_COMPOSE exec -e PGPASSWORD="$password" postgres psql -U "$user" -d "$db"
}

cmd_help() {
    echo ""
    echo "Engram Docker Setup"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  (default)     Start PostgreSQL with auto-detected port"
    echo "  --port PORT   Start with specific port"
    echo "  --down        Stop containers"
    echo "  --reset       Stop and delete all data"
    echo "  --status      Show status and stats"
    echo "  --logs        Follow container logs"
    echo "  --shell       Open PostgreSQL shell"
    echo "  --help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                     # Auto-detect port and start"
    echo "  $0 --port 5433         # Use port 5433"
    echo "  $0 --status            # Check status"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    case "${1:-}" in
        --down)
            cmd_down
            ;;
        --reset)
            cmd_reset
            ;;
        --status)
            cmd_status
            ;;
        --logs)
            cmd_logs
            ;;
        --shell)
            cmd_shell
            ;;
        --help|-h)
            cmd_help
            ;;
        --port)
            if [ -z "${2:-}" ]; then
                log_error "Port number required"
                exit 1
            fi
            cmd_up "$2"
            ;;
        "")
            cmd_up ""
            ;;
        *)
            log_error "Unknown command: $1"
            cmd_help
            exit 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
