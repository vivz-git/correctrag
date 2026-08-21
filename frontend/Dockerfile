# ─────────────────────────────────────────────────────────────────────────────
# CorrectRAG Frontend Dockerfile
# Lightweight Nginx Alpine web server for static browser UI
# ─────────────────────────────────────────────────────────────────────────────

FROM nginx:alpine

# Copy static frontend assets to standard Nginx html directory
COPY . /usr/share/nginx/html

# Expose HTTP port 80 (mapped to 3000 in docker-compose.yml)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
