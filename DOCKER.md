# Docker Development Guide

## Quick Start

```powershell
# 1. Build image (one-time)
docker build -f Dockerfile.dev -t pilotscope:dev `
  --build-arg enable_postgresql=true .

# 2. Start container with volume mount
docker run -it --rm `
  -v ${PWD}:/workspace `
  -w /workspace `
  pilotscope:dev bash

# 3. Inside container
conda activate pilotscope
python test_example_algorithms/simple_baseline.py

# 4. Edit code on host → Changes appear instantly!
```

## Using Docker Compose (Recommended)

```powershell
# Start
docker-compose up -d

# Enter container
docker-compose exec pilotscope-dev bash

# Inside
conda activate pilotscope
python test_example_algorithms/simple_baseline.py

# Stop
docker-compose down
```

## Why Two Dockerfiles?

- **`Dockerfile`** - Production: Clones from GitHub (for deployment)
- **`Dockerfile.dev`** - Development: Uses local files (for testing changes)

## Volume Mount Explained

```
Your PC (Windows)           Docker Container (Linux)
─────────────────          ───────────────────────
pilotscope/                /workspace/
  ├── pilotscope/    <──┐    ├── pilotscope/
  └── test_*.py         └────>  └── test_*.py
                        
Edit files here            Changes appear here instantly!
(VS Code)                 (No rebuild needed)
```

## When to Rebuild?

### ❌ No Rebuild Needed (99% of time)
- Editing Python code
- Adding new .py files
- Changing configurations

### ✅ Rebuild Required
- Adding packages to `requirements.txt`
- Changing `Dockerfile.dev`

```powershell
# Rebuild command
docker build -f Dockerfile.dev -t pilotscope:dev .
```

## Build Options

```powershell
# PostgreSQL only (faster)
docker build -f Dockerfile.dev -t pilotscope:dev `
  --build-arg enable_postgresql=true `
  --build-arg enable_spark=false .

# With Spark (slower, ~30min)
docker build -f Dockerfile.dev -t pilotscope:dev `
  --build-arg enable_postgresql=true `
  --build-arg enable_spark=true .
```

## Troubleshooting

### Port Already in Use
```powershell
# Use different port
docker run -it -v ${PWD}:/workspace -p 5433:5432 pilotscope:dev bash
```

### Changes Not Appearing
- Check volume mount: `-v ${PWD}:/workspace`
- Make sure you're editing files in project root
- Try restarting container

### Line Ending Errors (`\r` errors)
This is Windows (CRLF) vs Linux (LF) issue. Use simple commands:
```powershell
# ✅ Good
docker run -it -v ${PWD}:/workspace pilotscope:dev bash

# ❌ Avoid
docker run ... bash -c "multiple
lines
here"
```

## File Organization

Development files in project root:
```
pilotscope/
├── Dockerfile          # Production
├── Dockerfile.dev      # Development
├── docker-compose.yml  # Docker Compose config
├── .dockerignore       # Exclude files from build
└── DOCKER.md          # This file
```

This is standard practice - keeps Docker files accessible and visible.
