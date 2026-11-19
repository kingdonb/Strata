/** Qwen3.6 sizes are labelled through their family (sourceFamily 'qwen36'), never through the bare-name map, whose names are Flash-Next sets too. */
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
const compiled=await build({stdin:{contents:"export * from './src/lib/i18n'; export * from './src/lib/native-catalog';",resolveDir:process.cwd()},bundle:true,write:false,platform:'node',format:'esm'});
const {tr,setTranslationLanguage,nativeCatalogModels,catalogVersionLabel,loadCatalog}=await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const size=(id,model,extra={})=>({id,model,download_gb:4,ram_needed_gb:8,about:'',installed:false,...extra});
const family=(id,title,sizes)=>({family:id,title,images:true,sizes,about:'',experimental:false});
const grouped=()=>nativeCatalogModels([family('qwen36','Qwen3.6-35B-A3B',[size('qwen36-ud-q4_k_xl','UD-Q4_K_XL'),size('qwen36-ud-q6_k_xl','UD-Q6_K_XL'),size('qwen36-q8_0','Q8_0'),size('qwen36-new-size','NEW-SIZE')]),family('unsloth','Qwen3.8-Flash-Next Unsloth',[size('unsloth-ud-q4_k_xl','UD-Q4_K_XL')])]);
test('qwen36 sizes carry sourceFamily qwen36 after grouping',()=>{
 setTranslationLanguage('en');
 const q36=grouped().find(f=>f.family==='qwen3.6');
 assert.equal(q36.sizes.length,4);
 for(const v of q36.sizes)assert.equal(v.sourceFamily,'qwen36');
});
test('a qwen36 size shows its family label, the same bare name on Flash-Next stays plain',()=>{
 setTranslationLanguage('en');
 const groups=grouped();
 const q36=groups.find(f=>f.family==='qwen3.6');const unsloth=groups.find(f=>f.family==='qwen3.8');
 const byId=f=>Object.fromEntries(f.sizes.map(v=>[v.id,v]));
 assert.equal(catalogVersionLabel(byId(q36)['qwen36-ud-q4_k_xl']),'Smallest');
 assert.equal(catalogVersionLabel(byId(q36)['qwen36-ud-q6_k_xl']),'High quality');
 assert.equal(catalogVersionLabel(byId(q36)['qwen36-q8_0']),'Highest quality');
 assert.equal(catalogVersionLabel(byId(unsloth)['unsloth-ud-q4_k_xl']),'Unsloth','the edition carries it; the bare-name map is untouched');
});
test('a qwen36 size the labels do not know falls back to its plain name',()=>{
 setTranslationLanguage('en');
 const q36=grouped().find(f=>f.family==='qwen3.6');
 assert.equal(catalogVersionLabel(q36.sizes.find(v=>v.model==='NEW-SIZE')),'NEW-SIZE');
});
test('the labels translate where the string is a catalog key',async()=>{
 await loadCatalog('de');
 setTranslationLanguage('de');
 const q36=grouped().find(f=>f.family==='qwen3.6');
 const label=catalogVersionLabel(q36.sizes.find(v=>v.model==='UD-Q6_K_XL'));
 assert.notEqual(label,'High quality','High quality is a key (s0824) and translates');
 setTranslationLanguage('en');
 assert.equal(tr('High quality'),'High quality');
});
