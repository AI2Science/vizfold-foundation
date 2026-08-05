import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const REPO_ROOT = path.join(process.cwd(), "..");

export async function GET(request: NextRequest) {
  const filePath = request.nextUrl.searchParams.get("path");

  if (!filePath) {
    return new NextResponse("Missing path parameter", { status: 400 });
  }

  const absPath = path.join(REPO_ROOT, filePath);

  if (!fs.existsSync(absPath)) {
    return new NextResponse("File not found: " + absPath, { status: 404 });
  }

  const fileBuffer = fs.readFileSync(absPath);
  const ext = path.extname(absPath).toLowerCase();

  const contentTypes: Record<string, string> = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".pdb": "text/plain",
    ".txt": "text/plain",
    ".json": "application/json",
  };

  const contentType = contentTypes[ext] || "application/octet-stream";

  return new NextResponse(fileBuffer, {
    headers: { "Content-Type": contentType },
  });
}
