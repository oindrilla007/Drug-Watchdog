"use client";

import { useState, useRef, DragEvent, ChangeEvent } from "react";

type UploadMode = "prescription" | "pill_photo" | "lab_report" | "auto";

interface Props {
  onImage: (b64: string, mode: UploadMode) => void;
  loading?: boolean;
}

const MODE_LABELS: Record<UploadMode, string> = {
  auto:         "Auto-detect",
  prescription: "Prescription",
  pill_photo:   "Pill Photo",
  lab_report:   "Lab Report",
};

export default function ImageUploader({ onImage, loading }: Props) {
  const [mode, setMode]         = useState<UploadMode>("auto");
  const [preview, setPreview]   = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError]       = useState("");
  const inputRef                = useRef<HTMLInputElement>(null);

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, etc.)");
      return;
    }
    setError("");
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setPreview(result);
      // Strip data URL prefix: data:image/jpeg;base64,<actual_b64>
      const b64 = result.split(",")[1];
      onImage(b64, mode);
    };
    reader.readAsDataURL(file);
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div className="space-y-3">
      {/* Mode selector */}
      <div className="flex gap-2 flex-wrap">
        {(Object.keys(MODE_LABELS) as UploadMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`text-xs px-3 py-1.5 rounded-full border transition font-medium ${
              mode === m
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-300 hover:border-blue-400"
            }`}
          >
            {MODE_LABELS[m]}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative cursor-pointer rounded-xl border-2 border-dashed p-6 text-center transition
          ${dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"}
          ${loading ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onInputChange}
        />

        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Preview"
            className="max-h-48 mx-auto rounded-lg object-contain"
          />
        ) : (
          <div className="space-y-2">
            <div className="text-3xl">📷</div>
            <p className="text-sm text-gray-500">
              Drag &amp; drop an image, or <span className="text-blue-600 underline">browse</span>
            </p>
            <p className="text-xs text-gray-400">
              Prescription · Pill packets · Lab reports
            </p>
          </div>
        )}
      </div>

      {error && <p className="text-red-500 text-xs">{error}</p>}

      {preview && !loading && (
        <button
          onClick={() => { setPreview(null); if (inputRef.current) inputRef.current.value = ""; }}
          className="text-xs text-gray-500 underline"
        >
          Clear image
        </button>
      )}

      {loading && (
        <p className="text-center text-sm text-blue-600 animate-pulse">
          Scanning image…
        </p>
      )}
    </div>
  );
}
