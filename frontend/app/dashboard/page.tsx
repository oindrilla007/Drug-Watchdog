"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiGetProfile, apiGetAlerts, PatientProfile, AlertRecord } from "@/lib/api";
import LabValuesDisplay from "@/components/LabValuesDisplay";

export default function DashboardPage() {
  const [profile, setProfile]   = useState<PatientProfile | null>(null);
  const [alerts,  setAlerts]    = useState<AlertRecord[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error,   setError]     = useState("");

  useEffect(() => {
    Promise.all([apiGetProfile(), apiGetAlerts()])
      .then(([p, a]) => {
        setProfile(p);
        setAlerts(a.alerts);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
        {error}
        {error.includes("not found") && (
          <div className="mt-2">
            <Link href="/dashboard/profile" className="text-sm underline text-red-800">
              Complete your profile →
            </Link>
          </div>
        )}
      </div>
    );
  }

  const criticalAlerts = alerts.filter((a) => a.severity >= 3);
  const severeAlerts   = alerts.filter((a) => a.severity === 2);

  return (
    <div className="space-y-6">
      {/* Welcome */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          {profile ? `Hello, ${profile.name.split(" ")[0]}` : "Dashboard"}
        </h1>
        <p className="text-sm text-gray-500 mt-0.5">
          {profile?.last_session_at
            ? `Last check: ${new Date(profile.last_session_at).toLocaleDateString()}`
            : "No sessions yet"}
        </p>
      </div>

      {/* Alert summary banner */}
      {criticalAlerts.length > 0 && (
        <div className="rounded-xl border-2 border-red-200 bg-red-50 p-4 flex items-start gap-3">
          <span className="text-2xl">🔴</span>
          <div>
            <p className="font-semibold text-red-800">
              {criticalAlerts.length} critical alert{criticalAlerts.length !== 1 ? "s" : ""} need your attention
            </p>
            <Link href="/dashboard/alerts" className="text-sm text-red-700 underline mt-1 inline-block">
              View all alerts →
            </Link>
          </div>
        </div>
      )}

      {/* Quick actions */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Link
          href="/dashboard/analyse"
          className="rounded-xl border-2 border-blue-200 bg-blue-50 p-5 hover:bg-blue-100 transition group"
        >
          <div className="text-2xl mb-2">🔍</div>
          <h3 className="font-semibold text-blue-900">Check drugs</h3>
          <p className="text-sm text-blue-600 mt-0.5">
            Type drug names or scan a prescription
          </p>
        </Link>
        <Link
          href="/dashboard/alerts"
          className="rounded-xl border-2 border-gray-200 bg-white p-5 hover:bg-gray-50 transition"
        >
          <div className="text-2xl mb-2">🔔</div>
          <h3 className="font-semibold text-gray-900">
            Alerts{alerts.length > 0 && <span className="ml-2 text-xs bg-red-100 text-red-700 rounded-full px-2 py-0.5">{alerts.length}</span>}
          </h3>
          <p className="text-sm text-gray-500 mt-0.5">Unacknowledged interaction alerts</p>
        </Link>
        <Link
          href="/dashboard/profile"
          className="rounded-xl border-2 border-gray-200 bg-white p-5 hover:bg-gray-50 transition"
        >
          <div className="text-2xl mb-2">👤</div>
          <h3 className="font-semibold text-gray-900">Profile</h3>
          <p className="text-sm text-gray-500 mt-0.5">Update meds, labs, and conditions</p>
        </Link>
      </div>

      {/* Stats row */}
      {profile && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Medications", value: profile.medications.length },
            { label: "Conditions",  value: profile.conditions.length },
            { label: "Open alerts", value: alerts.length },
            { label: "Sessions",    value: profile.sessions_count },
          ].map((s) => (
            <div key={s.label} className="rounded-xl border border-gray-200 bg-white p-4 text-center">
              <p className="text-2xl font-bold text-gray-900">{s.value}</p>
              <p className="text-xs text-gray-400 mt-0.5">{s.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Clinical flags */}
      {profile && (profile.renal_impaired || profile.hepatic_impaired || profile.elderly || profile.high_bleed_risk) && (
        <div className="rounded-xl border border-orange-200 bg-orange-50 p-4">
          <h3 className="font-semibold text-orange-900 mb-2 text-sm">Active clinical flags</h3>
          <div className="flex flex-wrap gap-2">
            {profile.renal_impaired && <span className="text-xs bg-orange-100 text-orange-800 rounded-full px-3 py-1">Renal impairment (eGFR {profile.egfr?.toFixed(0)})</span>}
            {profile.hepatic_impaired && <span className="text-xs bg-orange-100 text-orange-800 rounded-full px-3 py-1">Hepatic impairment</span>}
            {profile.elderly && <span className="text-xs bg-orange-100 text-orange-800 rounded-full px-3 py-1">Elderly (age {profile.age})</span>}
            {profile.high_bleed_risk && <span className="text-xs bg-orange-100 text-orange-800 rounded-full px-3 py-1">High bleed risk (INR {profile.inr?.toFixed(1)})</span>}
          </div>
        </div>
      )}

      {/* Lab values */}
      {profile && (
        <div>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Lab values</h2>
          <LabValuesDisplay profile={profile} />
        </div>
      )}
    </div>
  );
}
