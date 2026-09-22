import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Drug Interaction Watchdog",
  description: "AI-powered drug interaction detection with patient-specific context",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900 antialiased">
        {children}
      </body>
    </html>
  );
}
