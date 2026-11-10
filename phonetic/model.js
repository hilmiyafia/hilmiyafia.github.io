let session;
let textbox;
const table = [
  '<start>', '<=>', '<,>', '<end>',
  "'", '-', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
  'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'AA0',
  'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1',
  'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D',
  'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
  'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH',
  'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1',
  'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'];
const ipa = {
  AA: "\u{251}", AE: "\u{E6}", AH: "\u{28C}", AO: "\u{254}", AW: "a\u{28A}",
  AY: "a\u{26A}", B: "b", CH: "t\u{283}", D: "d", DH: "\u{F0}",
  EH: "\u{25B}", ER: "\u{259}r", EY: "e\u{26A}", F: "f", G: "g",
  HH: "h", IH: "\u{26A}", IY: "i", JH: "d\u{292}", K: "k",
  L: "l", M: "m", N: "n", NG: "\u{14B}", OW: "o\u{28A}", OY: "\u{254}\u{26A}",
  P: "p", R: "r", S: "s", SH: "\u{283}", T: "t", TH: "\u{3B8}",
  UH: "\u{28A}", UW: "u", V: "v", W: "w", Y: "j", Z: "z", ZH: "\u{292}"
};

async function loadModel() {
  let prompt = document.getElementById("prompt");
  let button = document.getElementById("convert-button");
  prompt.textContent = "Please wait while the model is loading...";
  button.disabled = true;
  session = await ort.InferenceSession.create(
    "./model.onnx",
    { executionProviders: ["wasm"] }
  );
  prompt.textContent = "Write some text above then click the Convert button below!";
  button.disabled = false;
}

function encode(text) {
  let tokens = [];
  for (let i = 0; i < text.length; i++) {
    tokens.push(BigInt(table.indexOf(text[i])));
  }
  return tokens;
}

function decode(tokens) {
  let text = [];
  for (let i = 0; i < tokens.length; i++) {
    text.push(table[tokens[i]]);
  }
  return text;
}

async function sample(tokens) {
  let inputName = session.inputNames[0];
  let outputName = session.outputNames[0];
  for (let i = 0; i < 50; i++) {
    let tensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(tokens),
      [tokens.length]
    );
    let feeds = { [inputName]: tensor };
    let result = await session.run(feeds);
    let output = result[outputName];
    tokens.push(output.data);
    if (output.data == table.indexOf("<end>")) {
      break;
    }
  }
  return tokens;
}

async function runModel() {
  if (!session) {
    await loadModel();
  }
  if (!textbox) {
    textbox = document.getElementById("textbox");
  }
  let prompt = document.getElementById("prompt");
  let button = document.getElementById("convert-button");
  prompt.textContent = "Please wait while converting...";
  button.disabled = true;
  let lines = textbox.value.toLowerCase().replace(/[^A-Za-z]/g, " ").split("\n");
  let html = [];
  for (let k = 0; k < lines.length; k++) {
    let words = lines[k].split(" ");
    let line = [];
    for (let i = 0; i < words.length; i++) {
      if (words[i] == "") {
        continue;
      }
      let tokens = decode(await sample(encode([
        "<start>",
        ...words[i],
        "<=>"]
      )));
      let outputs = [[]];
      for (let j = 0; j < tokens.length; j++) {
        if (tokens[j] == "<start>" || tokens[j] == "<end>") {
          continue;
        }
        if (tokens[j] == "<=>") {
          outputs.push([]);
          continue;
        }
        if (tokens[j] == "<,>") {
          outputs.push([]);
          continue;
        }
        outputs[outputs.length - 1].push(tokens[j]);
      }
      let variants = [];
      for (let k = 1; k < outputs.length; k++) {
        for (let j = 0; j < outputs[k].length; j++) {
          outputs[k][j] = ipa[outputs[k][j].replace(/\d+/g, '')];
        }
        let variant = outputs[k].join("");
        if (variants.includes(variant) == false) {
          variants.push(variant);
        }
      }
      if (variants.length == 1) {
        line.push(variants[0]);
      } else {
        let numbered = [];
        for (let k = 0; k < variants.length; k++) {
          numbered.push((k + 1) + ". " + variants[k]);
        }
        line.push(
          "<span class=\"multi-pronunciation\">"
          + variants[0]
          + "<span class=\"popup\">"
          + numbered.join("<br>")
          + "</span></span>"
        );
      }
    }
    html.push(line.join(" "));
  }
  prompt.textContent = "Conversion finished!";
  button.disabled = false;
  document.getElementById("result").innerHTML = html.join("<br>");
}
loadModel();