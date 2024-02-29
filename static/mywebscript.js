function RunSentimentAnalysis(event) {
  console.log("Function called!");

  event.preventDefault(); // Prevent the form from submitting the traditional way

  const textToAnalyze = document.getElementById("textToAnalyze").value;

  console.log("Input Text:", textToAnalyze);
  const body = JSON.stringify({text: textToAnalyze});
  console.log("Request Body: ", body);

  fetch("/emotionDetector", {
    method: "POST",
    body,
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      const resultDiv = document.getElementById("system_response");
      console.log("Server Response:", data);
      resultDiv.innerHTML = `<p>Sentiment: ${data.result}</p>`;
    })
    .catch((error) => {
      console.error("Error:", error);
      // const resultDiv = document.getElementById("system_response");
      // resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    });

  return false; // Prevent the form from reloading the page
}
document.addEventListener("DOMContentLoaded", function() {
  const runButton = document.getElementById("runSentimentAnalysisButton");
  runButton.addEventListener("click", RunSentimentAnalysis);
});