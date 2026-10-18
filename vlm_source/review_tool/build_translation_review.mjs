import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const outputDir = path.join(root, "outputs");
const input = path.join(root, "data", "translation_candidates_pilot_01.jsonl");
const output = path.join(outputDir, "russian_translation_review_pilot_01.xlsx");
const previewPath = path.join(here, "translation_review_preview.png");

const rows = (await fs.readFile(input, "utf8"))
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

function renderSegments(targetText) {
  const target = JSON.parse(targetText);
  return (target.description_segments ?? []).map((segment) => {
    const top = Number.isFinite(segment.top_m) ? formatMeters(segment.top_m) : "?";
    const base = Number.isFinite(segment.base_m) ? formatMeters(segment.base_m) : "?";
    return `${top}–${base} м: ${segment.description ?? ""}`;
  }).join("\n\n");
}

function formatMeters(value) {
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function overallInterval(item) {
  const segments = JSON.parse(item.source_target).description_segments ?? [];
  const tops = segments.map((x) => x.top_m).filter(Number.isFinite);
  const bases = segments.map((x) => x.base_m).filter(Number.isFinite);
  if (!tops.length || !bases.length) return "";
  return `${formatMeters(Math.min(...tops))}–${formatMeters(Math.max(...bases))} м`;
}

function qaSummary(item) {
  const segments = JSON.parse(item.proposed_russian_target).description_segments ?? [];
  const descriptions = segments.map((segment) => segment.description ?? "");
  const joined = descriptions.join(" ");
  const repeated = descriptions.some((text) => /\b([\p{L}]{4,})\b(?:\s+\1\b){3,}/iu.test(text));
  const englishPattern = /\b(?:sand|mud|silt|clay|peat|shell|lag|fine|very|medium|coarse|root|organic|laminated|lamination|bioturbated|contact|gray|grey|brown|black|white|gap|sharp|gradational)\b/giu;
  const englishMatches = joined.match(englishPattern) ?? [];
  const qa = item.translation_qa ?? {};
  return [
    `Русский текст: ${qa.russian_text_in_every_segment ? "есть" : "ПРОВЕРИТЬ"}`,
    `Числа: ${qa.numeric_tokens_preserved ? "совпали" : "ПРОВЕРИТЬ"}`,
    `Глубины/сегменты: ${qa.depth_boundaries_preserved && qa.segment_count_preserved ? "совпали" : "ПРОВЕРИТЬ"}`,
    `Повтор текста: ${repeated ? "ПРОВЕРИТЬ" : "не найден"}`,
    `Англ. геотермины: ${englishMatches.length}`,
  ].join("\n");
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Проверка переводов");
sheet.showGridLines = false;
sheet.mergeCells("A1:H1");
sheet.mergeCells("A2:H2");
sheet.mergeCells("A3:H3");
sheet.getRange("A1").values = [["Проверка русских переводов описаний керна — 16 примеров"]];
sheet.getRange("A1:H1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 14, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
sheet.getRange("A1:H1").format.rowHeight = 30;
sheet.getRange("A2").values = [["Геологическое содержание исходных описаний подтверждено. Проверьте точность русского перевода; исходный текст слева, предложение локальной Qwen справа."]];
sheet.getRange("A3").values = [["Это черновые машинные переводы. Автопроверка отмечает глубины, числа, повторы и английские термины; она не заменяет проверку геологом. Исправления вносите в колонку H."]];
sheet.getRange("A2:H3").format = {
  font: { name: "Arial", size: 10, color: "#374151" },
  wrapText: true,
  verticalAlignment: "center",
};
sheet.getRange("A2:H2").format.rowHeight = 30;
sheet.getRange("A3:H3").format.rowHeight = 30;

const headers = [["№", "Набор / керн / ID", "Интервал", "Оригинал (английский)", "Предложенный перевод (русский)", "Автопроверка", "Решение", "Исправленный русский текст / комментарий"]];
sheet.getRange("A4:H4").values = headers;
const dataStart = 5;
const dataEnd = dataStart + rows.length - 1;
const values = rows.map((item, index) => [
  index + 1,
  `${item.dataset}\n${item.core_key}\n${item.record_id}`,
  overallInterval(item),
  renderSegments(item.source_target),
  renderSegments(item.proposed_russian_target),
  qaSummary(item),
  "Проверить перевод",
  "",
]);
sheet.getRange(`A${dataStart}:H${dataEnd}`).values = values;
sheet.tables.add(`A4:H${dataEnd}`, true, "TranslationReviewPilot01");
sheet.getRange("A4:H4").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sheet.getRange("A4:G4").format.rowHeight = 42;
sheet.getRange(`A${dataStart}:H${dataEnd}`).format = {
  font: { name: "Arial", size: 10, color: "#1F2937" },
  verticalAlignment: "top",
  wrapText: true,
};
sheet.getRange(`E${dataStart}:E${dataEnd}`).format.fill = "#FFF2CC";
sheet.getRange(`G${dataStart}:H${dataEnd}`).format.fill = "#FFF2CC";
sheet.getRange(`G${dataStart}:G${dataEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`A${dataStart}:A${dataEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`C${dataStart}:C${dataEnd}`).format.horizontalAlignment = "center";
sheet.getRange(`G${dataStart}:G${dataEnd}`).dataValidation = {
  rule: { type: "list", values: ["Проверить перевод", "Принято", "Исправить", "Отклонить"] },
};
const widths = { A: 6, B: 40, C: 16, D: 78, E: 78, F: 26, G: 20, H: 70 };
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}1:${column}${dataEnd}`).format.columnWidth = width;
}
sheet.getRange(`A${dataStart}:H${dataEnd}`).format.autofitRows();
sheet.getRange(`A${dataStart}:H${dataEnd}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  bottom: { style: "thin", color: "#D9E2F3" },
};
sheet.freezePanes.freezeRows(4);

workbook.recalculate();
const inspect = await workbook.inspect({
  kind: "region",
  sheetId: "Проверка переводов",
  range: "A1:H7",
  maxChars: 7000,
});
console.log(inspect.ndjson);
const preview = await workbook.render({ sheetName: "Проверка переводов", range: "A1:H6", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(output);
console.log(`created=${output}; rows=${rows.length}`);
