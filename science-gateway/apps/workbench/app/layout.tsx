import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VizFold",
  description: "Prototype multi-model protein structure visualization workbench"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
