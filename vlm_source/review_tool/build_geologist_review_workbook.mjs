import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const project = path.resolve(scriptDir, "..");
const outputDir = path.join(project, "outputs");
const outputPath = path.join(outputDir, "geologist_review_r41.xlsx");
const predictionsPath = path.join(
  outputDir,
  "core_vlm_v5_interval_crops_lora",
  "geologist_review_predictions.jsonl",
);
const thumbnailsDir = path.join(project, "tmp", "geologist_review_tool", "thumbnails");

const { SpreadsheetFile, Workbook } = await import("@oai/artifact-tool");

const predictionLines = (await fs.readFile(predictionsPath, "utf8"))
  .split(/\r?\n/)
  .filter(Boolean);
const predictions = predictionLines.map((line) => JSON.parse(line));
const ids = new Set(predictions.map((row) => row.record_id));
if (predictions.length !== 74 || ids.size !== 74) {
  throw new Error(`Expected 74 unique interval predictions, received ${predictions.length} rows and ${ids.size} IDs.`);
}
predictions.sort((a, b) => {
  const aTop = a.source_description_interval_m?.[0] ?? Number.POSITIVE_INFINITY;
  const bTop = b.source_description_interval_m?.[0] ?? Number.POSITIVE_INFINITY;
  if (aTop !== bTop) return aTop - bTop;
  return (a.source_row ?? 0) - (b.source_row ?? 0);
});
const lastRow = predictions.length + 1;
const validJsonCount = predictions.filter((row) => row.valid_json).length;

const workbook = Workbook.create();
const guide = workbook.worksheets.add("Как работать");
const review = workbook.worksheets.add("Проверка интервалов");
guide.showGridLines = false;
review.showGridLines = false;
guide.tabColor = "#7F9DB9";
review.tabColor = "#1F4E78";

guide.getRange("A1:B11").values = [
  ["Проверка черновиков описания керна", null],
  [null, null],
  ["Объём", "74 уникальные строки описаний из 57 фотографий скважины R-41. Один фотофрагмент представляет каждую строку; все 181 подтверждённые фотофрагмента уже включены в обучение."],
  ["1. Сверьте", "Сравните фото, эталонное первое предложение, полное исходное описание и черновик v5."],
  ["2. Выберите решение", "В столбце «Решение геолога» выберите: Не проверено, Принято, Исправить или Отклонить."],
  ["3. Исправьте текст", "Если черновик неточен, внесите подтверждённый вариант в «Описание после исправления». Не добавляйте признаки, которых нет в источнике или фото."],
  ["4. Оставьте комментарий", "Укажите причину исправления или сомнение в привязке фото к интервалу."],
  ["5. Отберите записи", "На листе «Проверка интервалов» включены фильтры по заголовкам. Отфильтруйте «Решение геолога», чтобы найти незаполненные или исправленные записи."],
  [null, null],
  ["Статус модели", `Черновики сгенерированы экспериментальной v5. В этом прогоне валидный JSON получен для ${validJsonCount} из 74 ответов; этот показатель проверяет только формат. Модель также добавляет неподтверждённые детали. Используйте текст только для сверки. v6 проверку не прошла и здесь не применяется.`],
  ["Следующий шаг", "Заполненные решения и исправленные описания можно использовать для подготовки следующей обучающей выборки после проверки геологом."],
];
guide.getRange("A1:B11").format.font = { name: "Arial", size: 11, color: "#222222" };
guide.getRange("A1:B1").format.font = { name: "Arial", size: 15, bold: true, color: "#17365D" };
guide.getRange("A3:A11").format.font = { name: "Arial", size: 10, bold: true, color: "#17365D" };
guide.getRange("A3:B11").format.wrapText = true;
guide.getRange("A3:B11").format.verticalAlignment = "center";
guide.getRange("A1:A11").format.columnWidthPx = 170;
guide.getRange("B1:B11").format.columnWidthPx = 760;
guide.getRange("A1:B1").format.rowHeightPx = 38;
guide.getRange("A3:B11").format.rowHeightPx = 48;
guide.getRange("A10:B10").format.rowHeightPx = 70;
guide.getRange("A11:B11").format.rowHeightPx = 52;

const headers = [
  "ID записи",
  "Фото фрагмента",
  "Верх фрагмента, м",
  "Низ фрагмента, м",
  "Верх исходного слоя, м",
  "Низ исходного слоя, м",
  "Эталон: первое предложение",
  "Исходное описание полностью",
  "Черновик модели v5",
  "JSON v5 корректен",
  "Решение геолога",
  "Описание после исправления",
  "Комментарий геолога",
  "Файл фотофрагмента",
  "Исходная таблица описаний",
  "Строка в исходной таблице",
];

const rows = predictions.map((p) => {
  let reference = p.reference_target_json ?? p.reference_first_sentence ?? "";
  try {
    const target = JSON.parse(reference);
    reference = target.description_segments?.[0]?.description ?? reference;
  } catch {
    // Keep raw target text visible if the training target is malformed.
  }
  const crop = Array.isArray(p.crop_interval_m) ? p.crop_interval_m : [null, null];
  const source = Array.isArray(p.source_description_interval_m)
    ? p.source_description_interval_m
    : [null, null];
  let draft = p.v5_draft_description ?? "";
  if (!draft && p.v5_output_raw) {
    const match = p.v5_output_raw.match(/"description"\s*:\s*"((?:\\.|[^"\\])*)/s);
    if (match) {
      try {
        draft = JSON.parse(`"${match[1]}"`);
      } catch {
        draft = match[1].replace(/\\n/g, " ").replace(/\\"/g, '"');
      }
    }
  }
  return [
    p.record_id,
    null,
    crop[0] ?? null,
    crop[1] ?? null,
    source[0] ?? null,
    source[1] ?? null,
    reference,
    p.source_description_full ?? "",
    draft,
    p.valid_json ? "Да" : "Нет",
    "Не проверено",
    "",
    "",
    p.image ?? "",
    p.source_workbook ?? "",
    p.source_row ?? null,
  ];
});

review.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
review.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows;
review.getRange(`A1:P${lastRow}`).format.font = { name: "Arial", size: 10, color: "#222222" };
review.getRange("A1:P1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
review.getRange("A1:P1").format.rowHeightPx = 42;
review.getRange(`A2:P${lastRow}`).format.verticalAlignment = "center";
review.getRange(`G2:I${lastRow}`).format.wrapText = true;
review.getRange(`L2:M${lastRow}`).format.wrapText = true;
review.getRange(`N2:P${lastRow}`).format.wrapText = true;
review.getRange(`A2:P${lastRow}`).format.rowHeightPx = 84;
review.getRange(`C2:F${lastRow}`).format.numberFormat = "0.00";
review.getRange(`K2:M${lastRow}`).format.fill = "#FFF2CC";

const widths = {
  A: 280,
  B: 198,
  C: 112,
  D: 112,
  E: 120,
  F: 120,
  G: 300,
  H: 430,
  I: 330,
  J: 112,
  K: 142,
  L: 330,
  M: 300,
  N: 340,
  O: 390,
  P: 145,
};
for (const [column, width] of Object.entries(widths)) {
  review.getRange(`${column}1:${column}${lastRow}`).format.columnWidthPx = width;
}

review.getRange(`K2:K${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["Не проверено", "Принято", "Исправить", "Отклонить"] },
};
review.getRange(`K2:K${lastRow}`).conditionalFormats.add("containsText", {
  text: "Исправить",
  format: { fill: "#FCE4D6", font: { color: "#9C0006", bold: true } },
});
review.getRange(`K2:K${lastRow}`).conditionalFormats.add("containsText", {
  text: "Принято",
  format: { fill: "#E2F0D9", font: { color: "#375623", bold: true } },
});
review.getRange(`K2:K${lastRow}`).conditionalFormats.add("containsText", {
  text: "Отклонить",
  format: { fill: "#E7E6E6", font: { color: "#595959", bold: true } },
});

const table = review.tables.add(`A1:P${lastRow}`, true, "GeologistReviewTable");
table.style = "TableStyleMedium2";
table.showFilterButton = true;
review.freezePanes.freezeRows(1);
review.freezePanes.freezeColumns(2);

for (let i = 0; i < predictions.length; i++) {
  const name = path.basename(predictions[i].image);
  const thumbPath = path.join(thumbnailsDir, name);
  const bytes = await fs.readFile(thumbPath);
  const dataUrl = `data:image/jpeg;base64,${bytes.toString("base64")}`;
  review.images.add({
    dataUrl,
    anchor: { from: { row: i + 1, col: 1 }, extent: { widthPx: 190, heightPx: 79 } },
  });
}

await fs.mkdir(outputDir, { recursive: true });
workbook.recalculate();
const guidePreview = await workbook.render({ sheetName: "Как работать", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(
  path.join(project, "tmp", "geologist_review_tool", "guide_preview.png"),
  new Uint8Array(await guidePreview.arrayBuffer()),
);
const reviewPreview = await workbook.render({
  sheetName: "Проверка интервалов",
  range: "A1:M5",
  scale: 0.8,
  format: "png",
});
await fs.writeFile(
  path.join(project, "tmp", "geologist_review_tool", "review_preview.png"),
  new Uint8Array(await reviewPreview.arrayBuffer()),
);
const inspect = await workbook.inspect({
  kind: "workbook,sheet,table,drawing",
  maxChars: 4500,
  tableMaxRows: 3,
  tableMaxCols: 8,
});
console.log(inspect.ndjson ?? inspect);
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
console.log(`Saved ${outputPath}; rows=${rows.length}; embedded_photos=${predictions.length}`);
