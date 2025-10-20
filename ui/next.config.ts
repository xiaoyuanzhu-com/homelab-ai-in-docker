import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  distDir: "dist",
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
