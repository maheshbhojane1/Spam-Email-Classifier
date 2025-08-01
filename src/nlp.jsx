export const preprocessText = (text) => {
  return text
    .toLowerCase()
    .replace(/[^a-z\s]/g, "")
    .split(" ")
    .filter((word) => word.length > 2); 
};

export const buildVocabulary = (emails) => {
  const vocab = new Set();
  emails.forEach((email) => preprocessText(email.text).forEach((word) => vocab.add(word)));
  return Array.from(vocab);
};

export const textToVector = (text, vocab) => {
  const words = preprocessText(text);
  return vocab.map((v) => words.filter((w) => w === v).length);
};
