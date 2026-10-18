import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "..");
const outputDir = path.join(projectRoot, "outputs");
const outputPath = path.join(outputDir, "geologist_review_pilot_01.xlsx");
const previewPath = path.join(here, "review_preview.png");

async function readJsonl(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function evenlySpaced(items, count) {
  if (items.length <= count) return [...items];
  if (count <= 1) return [items[Math.floor((items.length - 1) / 2)]];
  return Array.from({ length: count }, (_, i) => items[Math.round(i * (items.length - 1) / (count - 1))]);
}

function parseAnswer(row) {
  const raw = row.conversations?.find((message) => message.from === "gpt")?.value ?? "";
  try {
    const value = JSON.parse(raw);
    if (Array.isArray(value.description_segments)) return value.description_segments;
  } catch {
    // Keep the original candidate text below if the row is not valid JSON.
  }
  return [{ top_m: null, base_m: null, description: raw }];
}

function depthFor(row, pair) {
  const interval = pair?.image_depth_interval_m;
  if (Number.isFinite(interval?.top) && Number.isFinite(interval?.base)) {
    return [interval.top, interval.base];
  }
  const photoInterval = row.metadata?.photo_interval_m;
  if (Array.isArray(photoInterval) && photoInterval.every(Number.isFinite)) return photoInterval;
  const segments = parseAnswer(row).filter((segment) => Number.isFinite(segment.top_m) && Number.isFinite(segment.base_m));
  if (!segments.length) return [null, null];
  return [Math.min(...segments.map((segment) => segment.top_m)), Math.max(...segments.map((segment) => segment.base_m))];
}

function draftText(row) {
  const segments = parseAnswer(row);
  return segments.map((segment) => {
    const top = Number.isFinite(segment.top_m) ? `${segment.top_m}–` : "";
    const base = Number.isFinite(segment.base_m) ? `${segment.base_m} м` : "";
    const depth = top || base ? `${top}${base}: ` : "";
    return `${depth}${segment.description ?? ""}`.trim();
  }).join("\n");
}

const rows = await readJsonl(path.join(projectRoot, "data", "qwen3_vl_v2_external_experiment.jsonl"));
const pairs = await readJsonl(path.join(projectRoot, "data", "candidate_pairs.jsonl"));
const pairById = new Map(pairs.map((pair) => [pair.id, pair]));

const r41 = rows
  .filter((row) => !row.metadata?.dataset)
  .map((row) => ({ row, pair: pairById.get(row.metadata?.record_id) }))
  .filter(({ row, pair }) => pair && row.image && pair.image?.member)
  .sort((a, b) => (a.pair.image_depth_interval_m?.top ?? 0) - (b.pair.image_depth_interval_m?.top ?? 0));
const r41Unique = [];
const r41Seen = new Set();
for (const item of r41) {
  const key = item.pair.image.member;
  if (!r41Seen.has(key)) {
    r41Seen.add(key);
    r41Unique.push(item);
  }
}
const selected = evenlySpaced(r41Unique, 4);

const sourceNames = {
  usgs_2021_308_central_florida: "USGS 2021 · Central Florida",
  usgs_2021_312_fire_island: "USGS 2021 · Fire Island",
  usgs_2023_309_st_andrew_bay: "USGS 2023 · St Andrew Bay",
  usgs_2024_330_midnight_pass: "USGS 2024 · Midnight Pass",
};

for (const dataset of Object.keys(sourceNames)) {
  const available = rows
    .filter((row) => row.metadata?.dataset === dataset && row.image)
    .sort((a, b) => {
      const coreOrder = String(a.metadata?.core_key ?? "").localeCompare(String(b.metadata?.core_key ?? ""));
      if (coreOrder !== 0) return coreOrder;
      return (a.metadata?.photo_interval_m?.[0] ?? 0) - (b.metadata?.photo_interval_m?.[0] ?? 0);
    });

  const byCore = new Map();
  for (const row of available) {
    const key = row.metadata?.core_key ?? row.image;
    if (!byCore.has(key)) byCore.set(key, []);
    byCore.get(key).push(row);
  }

  const coreGroups = [...byCore.entries()];
  const chosenCores = evenlySpaced(coreGroups, Math.min(4, coreGroups.length));
  const groupRows = chosenCores.map(([, coreRows]) => coreRows[Math.floor((coreRows.length - 1) / 2)]);
  const chosenIds = new Set(groupRows.map((row) => row.metadata.record_id));
  const remaining = available.filter((row) => !chosenIds.has(row.metadata?.record_id));
  const fill = evenlySpaced(remaining, Math.max(0, 4 - groupRows.length));
  selected.push(...[...groupRows, ...fill].slice(0, 4).map((row) => ({ row, pair: null })));
}

const displayName = (dataset) => sourceNames[dataset] ?? "Р‑41 · Уренгой";
const outputRows = selected.map(({ row, pair }, index) => {
  const meta = row.metadata ?? {};
  const [top, base] = depthFor(row, pair);
  let sourcePath;
  let sourceReference;
  if (pair) {
    const provenance = pair.provenance ?? {};
    const firstSegment = pair.source_description_segments?.[0] ?? {};
    sourcePath = provenance.description_workbook ?? "";
    sourceReference = `Лист ${provenance.description_sheet ?? "Пр.8"}; строка ${firstSegment.source_row ?? ""}`;
  } else {
    sourcePath = meta.log_path ? path.resolve(projectRoot, meta.log_path) : "";
    sourceReference = `DOI ${meta.source_doi ?? ""}; OCR Qwen3-VL-2B, проверить по журналу`;
  }
  return {
    photoPath: row.image,
    sourcePath,
    values: [
      index + 1,
      displayName(meta.dataset),
      meta.core_key ?? "Р‑41",
      top,
      base,
      row.image,
      sourcePath,
      sourceReference,
      draftText(row),
      meta.dataset ? "Геология подтверждена; перевод нужен" : "Подтверждено; русский target",
      "",
      "",
      "",
      meta.dataset
        ? "Геологическое содержание подтверждено пользователем; русский перевод нужно проверить перед обучением."
        : "Подтверждено пользователем в чате 2026-10-07; target уже на русском.",
    ],
  };
});

const approvedDate = "2026-10-07";
const approvedRecords = selected.map(({ row }) => {
  const record = structuredClone(row);
  const target = record.conversations?.find((message) => message.from === "gpt")?.value ?? "";
  record.metadata = {
    ...(record.metadata ?? {}),
    geology_approved: true,
    review_status: record.metadata?.dataset
      ? "geology_approved_translation_pending"
      : "approved_by_user_chat_confirmation",
    training_approved: !record.metadata?.dataset,
    approved_by: "user",
    approved_at: approvedDate,
    approval_basis: record.metadata?.dataset
      ? "User confirmed geological content; original English target is not approved for Russian-language training until its translation is reviewed."
      : "User confirmed this Russian target as correct in chat.",
  };
  return {
    record,
    approval: {
      record_id: record.metadata.record_id,
      source_group: record.metadata.dataset ?? "well_R-41",
      decision: record.metadata.dataset ? "geology_approved_translation_pending" : "approved_unchanged",
      approved_by: "user",
      approved_at: approvedDate,
      basis: record.metadata.dataset
        ? "Explicit user confirmation in chat for geological content; Russian translation pending review."
        : "Explicit user confirmation in chat for the Russian target.",
      target_sha256: createHash("sha256").update(target, "utf8").digest("hex"),
    },
  };
});

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Проверка разметки");
sheet.showGridLines = false;

sheet.getRange("A1").values = [["Пилотная проверка разметки кернов — 20 примеров"]];
sheet.getRange("A1:N1").format.font = { name: "Arial", size: 14, bold: true, color: "#1F2937" };
sheet.getRange("A1:N1").format.rowHeight = 28;
sheet.getRange("A2").values = [["Геологическое содержание 20 строк подтверждено пользователем 07.10.2026; 4 русских target готовы, для 16 английских требуется проверить перевод."]];
sheet.getRange("A3").values = [["В обучение сейчас включены только 4 русских target Р‑41. Фация и обстановка не размечены отдельно."]];
sheet.getRange("A2:N3").format.font = { name: "Arial", size: 10, italic: true, color: "#4B5563" };
sheet.getRange("A2:N3").format.wrapText = false;

const headers = [[
  "№", "Источник / набор", "Керн", "Верх фото, м", "Низ фото, м", "Фото — полный путь",
  "Журнал / таблица — путь", "Источник текста / примечание", "Черновой ответ для проверки", "Решение",
  "Проверенный ответ для обучения", "Литология / фация", "Обстановка осадконакопления",
  "Комментарий геолога",
]];
sheet.getRange("A4:N4").values = headers;
const firstDataRow = 5;
const lastDataRow = firstDataRow + outputRows.length - 1;
sheet.getRange(`A${firstDataRow}:N${lastDataRow}`).values = outputRows.map((item) => item.values);

const reviewTable = sheet.tables.add(`A4:N${lastDataRow}`, true, "CoreReviewPilot01");
void reviewTable;
sheet.getRange(`A4:N4`).format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("A4:N4").format.rowHeight = 40;
sheet.getRange(`A5:N${lastDataRow}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
sheet.getRange(`A5:N${lastDataRow}`).format.verticalAlignment = "top";
sheet.getRange(`A5:N${lastDataRow}`).format.wrapText = true;
sheet.getRange(`K5:N${lastDataRow}`).format.fill = "#FFF2CC";
sheet.getRange(`J5:J${lastDataRow}`).format.fill = "#FCE4D6";
sheet.getRange(`D5:E${lastDataRow}`).format.numberFormat = "0.00";
sheet.getRange(`A5:A${lastDataRow}`).format.horizontalAlignment = "center";
sheet.getRange(`D5:E${lastDataRow}`).format.horizontalAlignment = "right";
sheet.getRange(`F5:G${lastDataRow}`).format.wrapText = false;
sheet.getRange(`H5:H${lastDataRow}`).format.wrapText = true;

const widths = {
  A: 6, B: 28, C: 15, D: 12, E: 12, F: 38, G: 40,
  H: 38, I: 66, J: 25, K: 66, L: 28, M: 34, N: 38,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}1:${column}${lastDataRow}`).format.columnWidth = width;
}

sheet.getRange(`J${firstDataRow}:J${lastDataRow}`).dataValidation = {
  rule: { type: "list", values: ["Подтвердить", "Исправить", "Отклонить", "Недостаточно данных"] },
};
sheet.getRange(`A5:N${lastDataRow}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  bottom: { style: "thin", color: "#D9E2F3" },
};
sheet.freezePanes.freezeRows(4);
sheet.getRange(`A5:N${lastDataRow}`).format.autofitRows();

workbook.recalculate();
const inspect = await workbook.inspect({
  kind: "region",
  sheetId: "Проверка разметки",
  range: `A1:N${Math.min(lastDataRow, 8)}`,
  maxChars: 5000,
});
console.log(inspect.ndjson);
const preview = await workbook.render({ sheetName: "Проверка разметки", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
await fs.writeFile(
  path.join(projectRoot, "data", "qwen3_vl_v3_reviewed_pilot.jsonl"),
  approvedRecords.map((item) => JSON.stringify(item.record)).join("\n") + "\n",
  "utf8",
);
await fs.writeFile(
  path.join(projectRoot, "data", "review_approvals_pilot_01.jsonl"),
  approvedRecords.map((item) => JSON.stringify(item.approval)).join("\n") + "\n",
  "utf8",
);
console.log(`created=${outputPath}`);
console.log(`rows=${outputRows.length}; approved_manifest=${approvedRecords.length}`);
