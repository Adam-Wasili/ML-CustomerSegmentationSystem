// Opens and closes the sidebar on small screens when the Menu button is pressed.
document.addEventListener("DOMContentLoaded", function () {
  var toggle = document.getElementById("sidebarToggle");
  var sidebar = document.getElementById("sidebar");
  // Both elements exist only for logged-in users, so check before using them
  if (toggle && sidebar) {
    toggle.addEventListener("click", function () {
      sidebar.classList.toggle("open");   // "open" is styled in style.css
    });
  }
});