export const API_BASE = "http://127.0.0.1:8000";

// Fetch system status from backend
export async function getSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/system/status`);
    if (res.ok) return await res.json();
  } catch (e) {
    // ignore
  }
  return null;
}

// Add other backend API helpers and utility functions here as needed
