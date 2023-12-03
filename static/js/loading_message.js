document.addEventListener("DOMContentLoaded", function() {
  // Add an event listener to the button
  var button = document.getElementById("downloadButton");
  button.addEventListener("click", function(event) {
      // Prevent the default form submission behavior
      event.preventDefault();

      // Disable the button to prevent multiple clicks
      button.disabled = true;

      // Show the loading message
      document.getElementById("loadingMessage").style.display = "block";

      // Your file processing logic here (if any)

      // Submit the form programmatically
      document.getElementById("downloadForm").submit();
  });
});


