import json
import sys

def parse_item(item, level=0):
    indent = "  " * level
    lines = []
    
    # 1. Header mit Metadaten (Pflichtfeld, Mehrfachauswahl)
    req_str = " (Pflichtfeld)" if item.get('required') else ""
    rep_str = " [Mehrfachauswahl]" if item.get('repeats') else ""
    lines.append(f"{indent}- **{item['linkId']}** (`{item['type']}`){req_str}{rep_str}")
    
    # 2. Fragetext
    if 'text' in item:
        lines.append(f"{indent}  - **Text:** {item['text']}")
    
    # 3. ValueSet Bindung
    if 'answerValueSet' in item:
        lines.append(f"{indent}  - **ValueSet:** `{item['answerValueSet']}`")
        
    # 4. FHIR Extensions parsen (Regex, SDC-Quantities)
    for ext in item.get('extension', []):
        url = ext.get('url', '')
        if 'regex' in url:
            lines.append(f"{indent}  - **Validierung (Regex):** `{ext.get('valueString')}`")
        elif 'minQuantity' in url:
            val = ext.get('valueQuantity', {}).get('value')
            unit = ext.get('valueQuantity', {}).get('unit', '')
            lines.append(f"{indent}  - **Min-Wert:** {val} {unit}")
        elif 'maxQuantity' in url:
            val = ext.get('valueQuantity', {}).get('value')
            unit = ext.get('valueQuantity', {}).get('unit', '')
            lines.append(f"{indent}  - **Max-Wert:** {val} {unit}")
            
    # 5. Logische Abhängigkeiten (enableWhen)
    if 'enableWhen' in item:
        conds = []
        for cond in item['enableWhen']:
            code_str = f" = `{cond['answerCoding']['code']}`" if 'answerCoding' in cond else f" {cond['operator']}"
            conds.append(f"`{cond['question']}`{code_str}")
        behavior = f" (Modus: {item.get('enableBehavior')})" if 'enableBehavior' in item else ""
        lines.append(f"{indent}  - **Sichtbar wenn:** {' AND '.join(conds)}{behavior}")
        
    # 6. Rekursion für Sub-Items (Kaskaden)
    if 'item' in item:
        for sub_item in item['item']:
            lines.extend(parse_item(sub_item, level + 1))
            
    return lines

def json_to_markdown(json_file_path, output_md_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    md = []
    md.append(f"# Implementation Guide: {data.get('title', 'Questionnaire')}\n")
    md.append(f"- **ID:** `{data.get('id')}`")
    md.append(f"- **URL:** {data.get('url')}")
    md.append(f"- **Version:** {data.get('version')}")
    md.append(f"- **Status:** {data.get('status')}\n")
    md.append(f"## Beschreibung\n{data.get('description', '')}\n")
    md.append("## Fragebogen-Struktur\n")
    
    for root_item in data.get('item', []):
        md.extend(parse_item(root_item, 0))
        
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    # Aufruf: python script.py questionnaire.json guide.md
    json_to_markdown(sys.argv[1], sys.argv[2])