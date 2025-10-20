import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  distDir: "dist",
  // Skip trailing slash to match FastAPI's behavior
  trailingSlash: true,
  // Disable static generation errors for dynamic routes
  // We'll handle 404s with FastAPI fallback to index.html
  skipTrailingSlashRedirect: true,
  // Only use rewrites in development mode
  // In production, FastAPI will serve the static files and handle API requests
  ...(process.env.NODE_ENV === "development" && {
    async rewrites() {
      return [
        {
          source: "/api/:path*",
          destination: "http://localhost:8000/api/:path*",
        },
      ];
    },
  }),
};

export default nextConfig;
