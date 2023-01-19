function accordionHandler(accordionElem, msgClick, msg1) {
  // var accordionElem = document.getElementsByClassName("accordion");
  var i;
  // for (i = 0; i < accordionElem.length; i++) {
  // accordionElem[i].addEventListener("click", function() {
  accordionElem.addEventListener("click", function() {
    /* Toggle between adding and removing the "active" class,
    to highlight the button that controls the panel */
    this.classList.toggle("active");

    /* Toggle between hiding and showing the active panel */
    var panel = this.nextElementSibling;

    /*  basic */
    // if (panel.style.display === "block") {
    //   panel.style.display = "none";
    // } else {
    //   panel.style.display = "block";
    // }

    // Animated Accordion (slide down)
    if (panel.style.maxHeight) {
      if (msgClick != undefined) {
        msgClick.innerHTML = msg1//"Show details"
      }
      
      panel.style.maxHeight = null;
    } else {
      if (msgClick != undefined) {
        msgClick.innerHTML = "Hide details"
      }
      panel.style.maxHeight = panel.scrollHeight + "px";
    }
  });
  // }
}

// var accordionElem = document.getElementsByClassName("accordion");
// console.log(typeof accordionElem);
// console.log(accordionElem.length);

function singleSelect() {
  var msgClick = document.getElementById("msg-button-SingleSelect");
  var msgOut = document.getElementById("msgSingleSelect"); 
  // create an ajax request
  $.ajax({
    type: "POST",
    url: "/select",
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
        // console.log("aaaa")
        msgOut.style.display = "block";
        // for accordion
        var default_msg = "Show options"
        msgClick.innerHTML = default_msg;
        accordionHandler(document.getElementById("accordion-SingleSelect"), msgClick, default_msg);
    }
  });
}
