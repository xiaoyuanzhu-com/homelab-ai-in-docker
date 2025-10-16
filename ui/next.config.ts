import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/api/status",
        destination: "http://localhost:8000/",
      },
      {
        source: "/health",
        destination: "http://localhost:8000/health",
      },
      {
        source: "/ready",
        destination: "http://localhost:8000/ready",
      },
      {
        source: "/docs",
        destination: "http://localhost:8000/docs",
      },
    ];
  },
};

export default nextConfig;
