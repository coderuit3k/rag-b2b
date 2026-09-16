"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartRow } from "@/lib/types";

// Cùng bộ 3 kiểu như app Streamlit (app.py::_render_chart) để nhất quán trải nghiệm.
const KINDS = ["Cột", "Tròn", "Đường"] as const;
type Kind = (typeof KINDS)[number];

const COLORS = ["#2563eb", "#0ea5e9", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"];

/** Cột đầu tiên KHÔNG phải số làm nhãn, cột số đầu tiên làm giá trị — khớp logic
 * tools/graph.py::_chartable() (mỗi dòng >=2 cột, có ít nhất 1 cột số). */
function pickColumns(rows: ChartRow[]) {
  const keys = Object.keys(rows[0] ?? {});
  const labelKey = keys.find((k) => typeof rows[0][k] !== "number") ?? keys[0];
  const valueKey = keys.find((k) => typeof rows[0][k] === "number");
  return { labelKey, valueKey };
}

export default function ChartRenderer({ data }: { data: ChartRow[] | null | undefined }) {
  const [kind, setKind] = useState<Kind>("Cột");
  if (!data || data.length === 0) return null;
  const { labelKey, valueKey } = pickColumns(data);
  if (!valueKey) return null;

  return (
    <div className="mt-3">
      <div className="mb-2 flex gap-1">
        {KINDS.map((k) => (
          <button
            key={k}
            onClick={() => setKind(k)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              kind === k ? "bg-blue-600 text-white" : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
            }`}
          >
            {k}
          </button>
        ))}
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {kind === "Cột" ? (
            <BarChart data={data} margin={{ left: 0, right: 12, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
              <XAxis dataKey={labelKey} tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey={valueKey} fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : kind === "Đường" ? (
            <LineChart data={data} margin={{ left: 0, right: 12, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
              <XAxis dataKey={labelKey} tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey={valueKey} stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          ) : (
            <PieChart>
              <Tooltip />
              <Pie data={data} dataKey={valueKey} nameKey={labelKey} outerRadius="80%" label>
                {data.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
            </PieChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
