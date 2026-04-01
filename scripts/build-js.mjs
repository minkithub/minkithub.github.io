import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { minify } from 'terser';

const sourcePath = fileURLToPath(new URL('../js/hux-blog.js', import.meta.url));
const outputPath = fileURLToPath(new URL('../js/hux-blog.min.js', import.meta.url));

const source = await readFile(sourcePath, 'utf8');
const result = await minify(source, {
  compress: true,
  mangle: true,
  format: {
    comments: false,
    preamble: `/*!
 * Hux Blog v1.7.0 (http://huxpro.github.io)
 * Copyright 2018 Hux <huxpro@gmail.com>
 */`,
  },
});

if (!result.code) {
  throw new Error('js minify 결과가 비어 있습니다.');
}

await writeFile(outputPath, `${result.code.trimEnd()}\n`);
