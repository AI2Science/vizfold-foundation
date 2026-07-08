import { HomePageClient } from "@/components/home-page-client";
import { executionTargets, modelOptions } from "@/lib/mock-data";

export default function HomePage() {
  return <HomePageClient modelOptions={modelOptions} executionTargets={executionTargets} />;
}

