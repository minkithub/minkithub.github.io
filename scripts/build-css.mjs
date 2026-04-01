import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import less from 'less';
import CleanCSSPlugin from 'less-plugin-clean-css';

const sourcePath = fileURLToPath(new URL('../less/hux-blog.less', import.meta.url));
const lessDir = fileURLToPath(new URL('../less/', import.meta.url));
const expandedPath = fileURLToPath(new URL('../css/hux-blog.css', import.meta.url));
const minifiedPath = fileURLToPath(new URL('../css/hux-blog.min.css', import.meta.url));

const source = await readFile(sourcePath, 'utf8');
const baseOptions = {
  filename: sourcePath,
  paths: [lessDir],
};

const expanded = await less.render(source, baseOptions);
const minified = await less.render(source, {
  ...baseOptions,
  plugins: [new CleanCSSPlugin({ advanced: true })],
});

await writeFile(expandedPath, `${expanded.css.trimEnd()}\n`);
await writeFile(minifiedPath, `${minified.css.trimEnd()}\n`);
