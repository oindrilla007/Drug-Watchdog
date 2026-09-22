"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import OnboardingForm from "@/components/OnboardingForm";

type Stage = "credentials" | "profile" | "done";

export default function SignupPage() {
  const router = useRouter();
  const [stage,    setStage]    = useState<Stage>("credentials");
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [confirm,  setConfirm]  = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");

  async function handleCredentials(e: React.FormEvent) {
    e.preventDefault();
    if (password !== confirm) { setError("Passwords do not match."); return; }
    if (password.length < 8)  { setError("Password must be at least 8 characters."); return; }
    setLoading(true);
    setError("");
    try {
      const { error: err } = await supabase.auth.signUp({ email, password });
      if (err) throw err;
      setStage("profile");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Signup failed");
    } finally {
      setLoading(false);
    }
  }

  if (stage === "profile") {
    return (
      <div className="min-h-screen bg-gray-50 py-10 px-4">
        <div className="max-w-lg mx-auto">
          <div className="text-center mb-8">
            <span className="text-4xl">💊</span>
            <h1 className="mt-2 text-2xl font-bold text-gray-900">Set up your profile</h1>
            <p className="text-sm text-gray-500 mt-1">
              This lets us personalise drug interaction severity for you.
            </p>
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <OnboardingForm onComplete={() => router.push("/dashboard")} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <span className="text-4xl">💊</span>
          <h1 className="mt-2 text-2xl font-bold text-gray-900">Create account</h1>
          <p className="text-sm text-gray-500 mt-1">Drug Watchdog — free forever</p>
        </div>

        <form onSubmit={handleCredentials} className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Min 8 characters"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Confirm password</label>
            <input
              type="password"
              required
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Repeat password"
            />
          </div>

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-blue-600 text-white font-semibold text-sm hover:bg-blue-700 disabled:opacity-40 transition"
          >
            {loading ? "Creating account…" : "Continue"}
          </button>
        </form>

        <p className="text-center text-sm text-gray-500 mt-4">
          Already have an account?{" "}
          <Link href="/auth/login" className="text-blue-600 underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
