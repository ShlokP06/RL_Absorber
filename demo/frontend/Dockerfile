# ── Adsorber — React Frontend (dev server) ────────────────────────────────────
#
# Uses Vite dev server — suitable for demos and local development.
# Not a production nginx build.
#
# Build:
#   docker build -t adsorber-frontend demo/frontend/
#
# Run standalone (backend must be on localhost:8000):
#   docker run -p 3000:3000 adsorber-frontend
#
# In Docker Compose the BACKEND_URL env var is set automatically.

FROM node:20-slim

WORKDIR /app

# Copy package files first — Docker caches this layer until package.json changes,
# so npm install only reruns when dependencies actually change.
COPY package.json package-lock.json* ./

# Install all dependencies (including devDependencies — needed for vite)
RUN npm install

# Copy the rest of the frontend source
COPY . .

EXPOSE 3000

# BACKEND_URL is injected by docker-compose so vite proxies to the correct host
CMD ["npm", "run", "dev"]
