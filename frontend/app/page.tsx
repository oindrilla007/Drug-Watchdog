import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <header className="border-b border-gray-200 bg-white px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-2xl">💊</span>
          <span className="font-bold text-gray-900 text-lg">Drug Watchdog</span>
        </div>
        <div className="flex gap-3">
          <Link href="/auth/login" className="text-sm text-gray-600 hover:text-gray-900 transition">
            Sign in
          </Link>
          <Link
            href="/auth/signup"
            className="text-sm px-4 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 transition"
          >
            Get started
          </Link>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center py-20">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 rounded-full bg-blue-50 border border-blue-100 px-4 py-1.5 text-xs text-blue-700 font-medium mb-6">
            <span>AI-powered pharmacovigilance</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 leading-tight mb-4">
            Catch dangerous drug<br />
            <span className="text-blue-600">interactions before they happen</span>
          </h1>
          <p className="text-lg text-gray-500 mb-8 leading-relaxed">
            Drug Watchdog combines XGBoost ML, medical RAG, and multi-agent AI to
            flag drug interactions — personalised for your kidney function, liver
            health, age, and comorbidities.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              href="/auth/signup"
              className="px-8 py-3 rounded-xl bg-blue-600 text-white font-semibold text-base hover:bg-blue-700 transition shadow-sm"
            >
              Create free account
            </Link>
            <Link
              href="/auth/login"
              className="px-8 py-3 rounded-xl border-2 border-gray-200 text-gray-700 font-semibold text-base hover:border-blue-300 transition"
            >
              Sign in
            </Link>
          </div>
        </div>

        {/* Feature cards */}
        <div className="mt-20 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl w-full text-left">
          {[
            {
              icon: "🔬",
              title: "ML + RAG Analysis",
              body: "XGBoost model trained on DrugBank, FAERS, and PubMed. RAG retrieval from 200k+ drug interaction documents.",
            },
            {
              icon: "📷",
              title: "Scan prescriptions",
              body: "Upload a photo of a prescription, pill packet, or lab report. Vision AI extracts the drugs automatically.",
            },
            {
              icon: "🧬",
              title: "Patient-specific context",
              body: "Severity is adjusted for your eGFR, liver enzymes, INR, age, pregnancy status, and prior alert history.",
            },
          ].map((f) => (
            <div key={f.title} className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <div className="text-2xl mb-2">{f.icon}</div>
              <h3 className="font-semibold text-gray-900 mb-1">{f.title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed">{f.body}</p>
            </div>
          ))}
        </div>
      </main>

      <footer className="border-t border-gray-200 px-6 py-4 text-center text-xs text-gray-400">
        Drug Watchdog — for educational and evaluation purposes. Not a substitute for professional medical advice.
      </footer>
    </div>
  );
}
