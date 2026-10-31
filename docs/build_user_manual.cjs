// Run with the bundled Codex Node runtime and its docx package.
const fs = require('fs');
const path = require('path');
const lib = process.env.CV1_DOCX_MODULE;
if (!lib) throw new Error('Set CV1_DOCX_MODULE to the bundled docx package');
const {Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, VerticalAlign,
  Header, Footer, PageNumber, LevelFormat} = require(lib);
const root = path.resolve(__dirname, '..');
const output = process.argv[2] || path.join(root, 'transfer/out/cv1-v2.1.0/CV1_User_Manual_RU.docx');
const lines = fs.readFileSync(path.join(__dirname, 'USER_GUIDE_RU.md'), 'utf8').split(/\r?\n/);
function runs(text, options={}) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
  return parts.filter(Boolean).map(value => new TextRun({text:value.replace(/^`|`$/g,'').replace(/^\*\*|\*\*$/g,''),
    bold:value.startsWith('**'), font:value.startsWith('`')?'Consolas':'Calibri', color:'000000', ...options}));
}
const children = [];
let inCode=false;
for(let i=0;i<lines.length;i++) {
  const text=lines[i];
  if(text.startsWith('```')) {inCode=!inCode;continue;}
  if(!text.trim()) continue;
  if(inCode) {
    // Commands remain editable and wrap naturally in Word.
    children.push(new Paragraph({children:[new TextRun({text,font:'Consolas',size:19,color:'000000'})],
      spacing:{before:30,after:90,line:250},keepLines:true}));continue;
  }
  if(text.startsWith('# ')) {children.push(new Paragraph({heading:HeadingLevel.TITLE,children:runs(text.slice(2))}));continue;}
  if(text.startsWith('## ')) {children.push(new Paragraph({heading:HeadingLevel.HEADING_1,children:runs(text.slice(3))}));continue;}
  if(text.startsWith('### ')) {children.push(new Paragraph({heading:HeadingLevel.HEADING_2,children:runs(text.slice(4))}));continue;}
  if(text.startsWith('|')) {
    const records=[];
    while(i<lines.length && lines[i].startsWith('|')) {
      const row=lines[i].split('|').slice(1,-1).map(v=>v.trim());
      if(!row.every(v=>/^[-: ]+$/.test(v))) records.push(row);
      i++;
    }
    i--;
    const n=records[0].length;
    const widths=n===3?[2450,1050,5860]:[3000,6360];
    const border={style:BorderStyle.SINGLE,size:4,color:'D9D9D9'};
    children.push(new Table({width:{size:9360,type:WidthType.DXA},columnWidths:widths,
      borders:{top:border,bottom:border,left:border,right:border,insideHorizontal:border,insideVertical:border},
      rows:records.map((record,index)=>new TableRow({tableHeader:index===0,cantSplit:true,
        children:record.map((value,col)=>new TableCell({width:{size:widths[col],type:WidthType.DXA},
          verticalAlign:VerticalAlign.CENTER,margins:{top:100,bottom:100,left:120,right:120},
          shading:{fill:index===0?'E7EDF2':'FFFFFF'},
          children:[new Paragraph({children:runs(value,{size:21,bold:index===0}),
            alignment:n===3&&col===1?AlignmentType.CENTER:AlignmentType.LEFT,
            spacing:{after:0,line:265}})]}))}))}));
    children.push(new Paragraph({children:[],spacing:{after:100}}));continue;
  }
  const numbered=text.match(/^(\d+)\. (.*)/);
  if(numbered) {children.push(new Paragraph({children:runs(text),indent:{left:240,hanging:240},spacing:{after:110,line:275}}));continue;}
  if(text.startsWith('- ')) {children.push(new Paragraph({children:runs(text.slice(2)),bullet:{level:0},spacing:{after:100,line:275}}));continue;}
  children.push(new Paragraph({children:runs(text),spacing:{after:130,line:280},widowControl:true}));
}
const document = new Document({creator:'CV-1',title:'Руководство пользователя CV 1',description:'Инструкция геологу и администратору версии 2.1.0',
  styles:{default:{document:{run:{font:'Calibri',size:23,color:'000000'},paragraph:{spacing:{after:130,line:280}}}},
    paragraphStyles:[
      {id:'Title',name:'Title',basedOn:'Normal',next:'Normal',quickFormat:true,
        run:{font:'Calibri',size:46,bold:true,color:'000000'},paragraph:{spacing:{before:0,after:250},keepNext:true}},
      {id:'Heading1',name:'heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,
        run:{font:'Calibri',size:30,bold:true,color:'000000'},paragraph:{spacing:{before:280,after:150},keepNext:true}},
      {id:'Heading2',name:'heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,
        run:{font:'Calibri',size:26,bold:true,color:'000000'},paragraph:{spacing:{before:210,after:120},keepNext:true}}
    ]},
  sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:1000,bottom:1000,left:1440,right:1440,header:420,footer:420}}},
    headers:{default:new Header({children:[new Paragraph({children:[new TextRun({text:'CV 1  ·  Руководство пользователя  ·  Версия 2.1.0',size:18,color:'000000'})],spacing:{after:0}})]})},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.RIGHT,children:[
      new TextRun({text:'Страница ',size:18,color:'000000'}),new TextRun({children:[PageNumber.CURRENT],size:18,color:'000000'})]})]})},children}]});
fs.mkdirSync(path.dirname(output),{recursive:true});
Packer.toBuffer(document).then(buffer=>{fs.writeFileSync(output,buffer);console.log(output,buffer.length);});
