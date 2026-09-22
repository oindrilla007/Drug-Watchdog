"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";

const NAV = [
  { href: "/dashboard",          label: "Dashboard",    icon: "🏠" },
  { href: "/dashboard/analyse",  label: "Drug Check",   icon: "🔍" },
  { href: "/dashboard/alerts",   label: "Alerts",       icon: "🔔" },
  { href: "/dashboard/profile",  label: "Profile",      icon: "👤" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router   = useRouter();
  const pathname = usePathname();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      if (!data.session) router.replace("/auth/login");
      else setChecking(false);
    });
  }, [router]);

  async function handleLogout() {
    await supabase.auth.signOut();
    router.push("/");
  }

  if (checking) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top nav */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xl">💊</span>
          <span className="font-bold text-gray-900 hidden sm:block">Drug Watchdog</span>
        </div>
        <nav className="flex gap-1 sm:gap-2">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={`flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition ${
                pathname === n.href
                  ? "bg-blue-50 text-blue-700"
                  : "text-gray-500 hover:text-gray-900 hover:bg-gray-100"
              }`}
            >
              <span>{n.icon}</span>
              <span className="hidden sm:block">{n.label}</span>
            </Link>
          ))}
        </nav>
        <button
          onClick={handleLogout}
          className="text-xs text-gray-400 hover:text-gray-600 transition"
        >
          Sign out
        </button>
      </header>

      {/* Page content */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-6">
        {children}
      </main>
    </div>
  );
}
