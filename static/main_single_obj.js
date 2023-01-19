function objUpload() {
  var fileInput = document.getElementById("file");
  
  var fileN = fileInput.value.split("\\");
  var fileName = fileN[fileN.length-1];
  var file = fileInput.files[0];
  var ins = fileInput.files.length
  var msg = ""
  if(ins == 0) {
    msg = 'Select at least one file'; //$('#msg').html('<span style="color:red">Select at least one file</span>');
    return;
  }
  var form_data = new FormData();
  for (var x = 0; x < ins; x++) {
    form_data.append("files[]", fileInput.files[x]);
  }

  // console.log(fileInput.files.length)
  // // Display the values
  // for (const value of form_data.values()) {
  //   console.log(value);
  // }

  var msgFile = document.getElementById("msg-file");
  var msgFileOut = document.getElementById("msg-upload");
  

  // store the data to be pass back to the server in an array
  // var server_data = [{
  //   "filename": fileName,
  //   "files": form_data
  // }];
  form_data.append("filename[]", fileName);

  // msgFileOut.style.display = "block";
  msgFile.innerHTML = fileName;

  // create an ajax request
  $.ajax({
    type: "POST",
    url: "/single_upload_obj",
    data: form_data, //JSON.stringify(server_data),
    contentType: false,
    processData: false,
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
          msgFileOut.style.display = "block";
          // console.log("aaaa")
    }
  });
}


function objRead() {
  var savedir = document.getElementById("save-dir");
  var msgOut  = document.getElementById("msg-read");
  var markers = document.getElementById("markers-ori");
  var channels = document.getElementById("channels-ori");
  var msgClick = document.getElementById("msg-button-save");

  var selectRemoveChannels = document.getElementById("selectRemoveChannels");
  var selectDefineChannels = document.getElementById("selectDefineChannels");


  // create an ajax request
  $.ajax({
    type: "POST",
    url: "/single_read_obj",
    // data: JSON.stringify(server_data),
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
        markers.innerHTML  = result.markers;
        channels.innerHTML = result.channels;
        savedir.innerHTML  = result.savedir;
        msgOut.style.display = "block";
        
        msgClick.innerHTML = "Show details"
        accordionHandler(document.getElementById("accordion-1"), msgClick, "Show details") 


        const channelsOri = result.channels.split(', ');
        for (i in channelsOri) {
          const channel = channelsOri[i]

          // create option using DOM
          const option     = document.createElement("option");
          const optionText = document.createTextNode(channel);

          // set option text and value
          option.appendChild(optionText)
          option.setAttribute("value", channel)
          
          selectRemoveChannels.appendChild(option);

          // const option1     = document.createElement("option");
          const option1 = option.cloneNode(true)
          // option1.appendChild(optionText)
          // option1.setAttribute("value", channel)
          selectDefineChannels.appendChild(option1);
        }
    }
  })
}
