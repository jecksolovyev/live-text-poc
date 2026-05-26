import express from 'express';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const CACHE_DIR = '.cache';
if (!existsSync(CACHE_DIR)) await mkdir(CACHE_DIR);

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(express.static('public'));

app.post('/ocr', async (req, res) => {
  const { imageUrl, languageHints, apiKey } = req.body;
  if (!apiKey) return res.status(400).json({ error: 'apiKey required' });
  if (!imageUrl) return res.status(400).json({ error: 'imageUrl required' });

  const cacheKey = createHash('sha256').update(imageUrl).digest('hex');
  const cachePath = path.join(CACHE_DIR, `${cacheKey}.json`);

  if (existsSync(cachePath)) {
    const cached = JSON.parse(await readFile(cachePath, 'utf8'));
    return res.json({ ...cached, cached: true });
  }

  const visionRes = await fetch(
    `https://vision.googleapis.com/v1/images:annotate?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        requests: [{
          image: { source: { imageUri: imageUrl } },
          features: [{ type: 'DOCUMENT_TEXT_DETECTION' }],
          imageContext: { languageHints: languageHints || ['uk', 'en'] }
        }]
      })
    }
  );

  const visionData = await visionRes.json();
  if (visionData.error) {
    return res.status(visionRes.status || 500).json({ error: visionData.error });
  }
  const annotation = visionData.responses?.[0];
  if (annotation?.error) {
    return res.status(500).json({ error: annotation.error });
  }

  const words = extractWords(annotation);
  const result = { words };
  await writeFile(cachePath, JSON.stringify(result));
  res.json(result);
});

function extractWords(annotation) {
  const words = [];
  const pages = annotation?.fullTextAnnotation?.pages || [];
  for (const page of pages) {
    for (const block of page.blocks || []) {
      for (const para of block.paragraphs || []) {
        for (const word of para.words || []) {
          const symbols = word.symbols || [];
          const text = symbols.map(s => s.text).join('');
          const verts = word.boundingBox?.vertices || [];
          if (verts.length < 4 || !text) continue;
          const xs = verts.map(v => v.x || 0);
          const ys = verts.map(v => v.y || 0);
          const x = Math.min(...xs);
          const y = Math.min(...ys);
          const w = Math.max(...xs) - x;
          const h = Math.max(...ys) - y;
          if ((word.confidence ?? 1) < 0.5) continue;
          const breakType = symbols[symbols.length - 1]?.property?.detectedBreak?.type;
          const trail = breakType === 'LINE_BREAK' || breakType === 'EOL_SURE_SPACE'
            ? '\n'
            : breakType ? ' ' : '';
          words.push({ text: text + trail, x, y, w, h });
        }
      }
    }
  }
  return words;
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`POC running at http://localhost:${PORT}`));
