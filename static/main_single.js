function dirSet() {
  var dirOut = document.getElementById("directory");
  var msgDir = document.getElementById("msg-dir");
  var msgSetOut = document.getElementById("msg-set");

  // msgSetOut.style.display = "block";

  // store the data to be pass back to the server in an array
  var server_data = [
    {
    "dir_out": dirOut.value}
  ];
  msgDir.innerHTML = dirOut.value;
  

  // create an ajax request
  $.ajax({
    type: "POST",
    url: "/single_set",
    data: JSON.stringify(server_data),
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
//        console.log("aaaa");
        msgSetOut.style.display = "block";
    }
  });
}

function fileUpload() {
  var fileInput = document.getElementById("file");
  var file = fileInput.value.split("\\");
  var fileName = file[file.length-1];
  var msgFile = document.getElementById("msg-file");
  var msgFileOut = document.getElementById("msg-upload");
  

  // store the data to be pass back to the server in an array
  var server_data = [
    {
    "filename": fileName}
  ];

  // msgFileOut.style.display = "block";
  msgFile.innerHTML = fileName;

  // create an ajax request
  $.ajax({
    type: "POST",
    url: "/single_upload",
    data: JSON.stringify(server_data),
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
          msgFileOut.style.display = "block";
          // console.log("aaaa")
    }
  });
}

function dataRead() {
  var sid = document.getElementById("slide");
  var rid = document.getElementById("roi");
  var savedir = document.getElementById("save-dir");
  var msgReadOut = document.getElementById("msg-read");
  var markers = document.getElementById("markers-ori");
  var channels = document.getElementById("channels-ori");
  var msgClick = document.getElementById("msg-button-save");

  var selectRemoveChannels = document.getElementById("selectRemoveChannels");
  var selectDefineChannels = document.getElementById("selectDefineChannels");
  
  
  // store the data to be pass back to the server in an array
  var server_data = [
    {
      "slide": sid.value,
      "roi": rid.value
    }
  ];

  // create an ajax request
  $.ajax({
    type: "POST",
    url: "/single_read",
    data: JSON.stringify(server_data),
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
        markers.innerHTML  = result.markers;
        channels.innerHTML = result.channels;
        savedir.innerHTML  = result.savedir;
        msgReadOut.style.display = "block";
        

        // var accordionElem = document.getElementsByClassName("accordion");
        // console.log(document.getElementsByClassName("accordion").length)
        // var accordionElem1 = document.getElementById("accordion-1");
        // console.log(typeof document.getElementById("accordion-1"))
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

function channelCheck() {
  var saveChannels = document.getElementById("save-channels");
  var msgOut = document.getElementById("msg-check-channels");
  var img = document.getElementById("fig-channels");
  var msgClick = document.getElementById("msg-button-check");

  $.ajax({
    type: "POST",
    url: "/single_check",
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true')
        // console.log(result.savedir)
        saveChannels.innerHTML = result.savedir;
        
        msgOut.style.display = "block";
        // console.log(result.img_data)
        img.src = 'data:image/png;base64,'+result.img_data.split("\'")[1]; //result.savedir; //
        
        // var accordionElem = document.getElementsByClassName("accordion");
        msgClick.innerHTML = "Show details"
        accordionHandler(document.getElementById("accordion-2"), msgClick, "Show details")
    }
  })
}

function channelRemove() {
  var selectRemoveChannels = document.getElementById("selectRemoveChannels");
  var msgOut   = document.getElementById("msgRemove");
  var channelsRemove = document.getElementById("channelsRemove");
  var img      = document.getElementById("fig-channels");
  var msgClick = document.getElementById("msg-button-rm");
  var selected = [];
  for (var option of selectRemoveChannels.options) {
    if (option.selected) {
      selected.push(option.value)
    }
  }
  // for (i in JSON.stringify(selectDefineChannels.options)) {
  //   option = selectDefineChannels.options[i];
  //   console.log(option.value);
  // }
  
  var server_data = [            // store the data to be pass back to the server in an array
    {
      "selected": selected
    }
  ];

  $.ajax({
    type: "POST",
    url: "/single_remove",
    contentType: "application/json",
    dataType: 'json',
    data: JSON.stringify(server_data),
    success: function(result) {
      if (result.processed == 'true')
        // update channels plots
        img.src = 'data:image/png;base64,'+result.img_data.split("\'")[1]; 

        // for accordion
        msgClick.innerHTML = "Show details"
        accordionHandler(document.getElementById("accordion-rm"), msgClick, "Show details")

        // show messages (make visible)
        channelsRemove.innerHTML = result.channels // JSON.stringify(selected) 
        msgOut.style.display = "block";

        // update dropdown list
        var selectDefineChannels = document.getElementById("selectDefineChannels");
        const channelsOri = result.channels.split(', ');
        
        while (selectDefineChannels.options.length > 0) {
          selectDefineChannels.remove(0);
        }  

        for (i in channelsOri){
          const channel = channelsOri[i];
        // for (i in JSON.stringify(selected)) {
          // const channel = JSON.stringify(selected)[i];
          // console.log(channel)
          // console.log(channel.options)
          // selectDefineChannels.option[value=channel].remove();

          // create option using DOM
          const option     = document.createElement("option");
          const optionText = document.createTextNode(channel);

          // set option text
          option.appendChild(optionText)

          // set option value
          option.setAttribute("value", channel)
          
          selectDefineChannels.appendChild(option);
        }
    }
  })
}

function channelDefineAdd() {
  var selectDefineChannels = document.getElementById("selectDefineChannels");
  var newName = document.getElementById("newChannel");
  var selected = [];
  for (var option of selectDefineChannels.options) {
    if (option.selected) {
      selected.push(option.value)
    }
  }

  // store the data to be pass back to the server in an array
  var server_data = [
    {
      "newName": newName.value,
      "oldNames": selected
    }
  ];

  $.ajax({
    type: "POST",
    url: "/single_define_add",
    contentType: "application/json",
    dataType: 'json',
    data: JSON.stringify(server_data), 
  })
}

function channelDefineFinish() {
  var msgOut   = document.getElementById("msgDefine");
  var channelsDefine = document.getElementById("channelsDefine");
  var msgClick = document.getElementById("msg-button-df");
  $.ajax({
    type: "POST",
    url: "/single_define_finish",
    contentType: "application/json",
    dataType: 'json',
    // data: JSON.stringify(server_data),
    success: function(result) {
      if (result.processed == 'true') 
        // for accordion
        msgClick.innerHTML = "Show details"
        accordionHandler(document.getElementById("accordion-df"), msgClick, "Show details")

        // console.log(result.channels)
        msgOut.style.display = "block";
        channelsDefine.innerHTML = result.channels; // JSON.stringify(selected) }
      
    }
  })
}

function cellSeg() {
  var cellSegR = document.getElementById("cellSegR");
  var img      = document.getElementById("fig-seg");
  var msgClick = document.getElementById("msg-button-seg");
  var msgOut   = document.getElementById("msgSeg");
  
  var server_data = [
    {
    "radius": cellSegR.value}
  ];
  $.ajax({
    type: "POST",
    url: "/single_cell_seg",
    contentType: "application/json",
    data: JSON.stringify(server_data),
    dataType: 'json',
    success: function(result) {
      if (result.processed == 'true') 
        // plot
        img.src = 'data:image/png;base64,'+result.img_data.split("\'")[1]; 
        // for accordion
        msgClick.innerHTML = "Show segmentation visualization";
        accordionHandler(document.getElementById("accordion-seg"), msgClick, "Show segmentation visualization");
        msgOut.style.display = "block";
    }
  })
}

function featExtract() {
  var msgOut = document.getElementById("msgExtract");
  $.ajax({
    type: "POST",
    url: "/single_feat_extract",
    contentType: "application/json",
    // dataType: 'json',
    success: function(result) {
      if (result.processed == 'true') 
        msgOut.style.display = "block";
    }
  })
}

function loadObj() {
  var msgOut = document.getElementById("msgLoadObj");
  $.ajax({
    type: "POST",
    url: "/single_load_obj",
    contentType: "application/json",
    success: function(result) {
      if (result.processed == 'true') 
        msgOut.style.display = "block";
    }
  })
}

function PhenoGraph() {
  var msgOut     = document.getElementById("msgPG");
  var msgClick   = document.getElementById("msg-button-PG");
  var img        = document.getElementById("fig-PG");
  var numCluster = document.getElementById("n-PG");
  var colorTable = document.getElementById("fig-colortable")
  var visPgOg    = document.getElementById("fig-PG-og")

  $.ajax({
    type: "POST",
    url: "/single_PhenoGraph",
    contentType: "application/json",
    success: function(result) {
      if (result.processed == 'true') 
        numCluster.innerHTML = result.n_cluster;

        // plots
        img.src = 'data:image/png;base64,'+result.img_data.split("\'")[1]; 

        colorTable.src = 'data:image/png;base64,'+result.color_table.split("\'")[1];
        visPgOg.src    = 'data:image/png;base64,'+result.PG_og.split("\'")[1];
        
        // for accordion
        var default_msg = "Show PhenoGraph clustering visualization"
        msgClick.innerHTML = default_msg;
        accordionHandler(document.getElementById("accordion-PG"), msgClick, default_msg);
        
        msgOut.style.display = "block";
    }
  })
}

function showDropdown() {
  // document.getElementById("markerPosDropDown").classList.toggle("show"); // show the dropdown menu
  var markerPosDropDown = document.getElementById("markerPosDropDown");
  var selectMPmarker = document.getElementById("selectMPmarker");
  
  // get dropdown contents
  $.ajax({
    type: "POST",
    url: "/single_MarkerPosPre",
    contentType: "application/json",
    success: function(result) {
      if (result.processed == 'true') {
        if (selectMPmarker.children.length <= 0) {
          channels = result.channels.split(", ");
          markers  = result.channels.split(", ");
          for (i in markers) {
            const chn = markers[i];
              // create option using DOM
              const option = document.createElement("option");
              const optionText = document.createTextNode(chn);
              // set option text
              option.appendChild(optionText)
              // set option value
              option.setAttribute("value", chn)
              selectMPmarker.appendChild(option);
          }
        }
        if (markerPosDropDown.style.display !== "block") {
            markerPosDropDown.style.display = "block";
            console.log(markerPosDropDown.style.display === "block")
        } else {
            markerPosDropDown.style.display = "none";
            console.log(markerPosDropDown.style.display === "block")
        }

      }
     }
  })
}

// Also close the dropdown menu if the user clicks outside of it
window.onclick = function(event) {
    console.log(event.target);
    if(!event.target.matches('.dropbtn')) {
        var openDropdown = document.getElementById("markerPosDropDown");
        console.log(openDropdown.style.display)
        if (openDropdown.style.display === "block") {
            openDropdown.style.display = "none";
        }
    }
}


function filterFunction() {
  var input, filter, ul, li, a, i;
  input = document.getElementById("myInput");
  filter = input.value.toUpperCase();
  div = document.getElementById("markerPosDropDown");

  // console.log(typeof div);
  // console.log(div.options)

  a = div.getElementsByTagName("p");
  console.log(a)

  for (i = 0; i < a.length; i++) {
    txtValue = a[i].textContent || a[i].innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      a[i].style.display = "";
    } else {
      a[i].style.display = "none";
    }
  }
}

function MarkerPos() {
  var msgOut   = document.getElementById("msgMarkerPos");
  var msgClick = document.getElementById("msg-button-MP");
  var visMP    = document.getElementById("fig-MP")
  var selectMPmarker = document.getElementById("selectMPmarker");
//  get selected marker
  var selected = [];
  for (var option of selectMPmarker.options) {
    if (option.selected) {
      selected.push(option.value)
    }
  }
  console.log(selected)
  var server_data = [{"selected": selected}];

  $.ajax({
    type: "POST",
    url: "/single_MarkerPos",
    contentType: "application/json",
    dataType: 'json',
    data: JSON.stringify(server_data),
    success: function(result) {
        // plots
        visMP.src    = 'data:image/png;base64,'+result.vis_marker_pos.split("\'")[1];
        
        // for accordion
        var default_msg = "Show marker positive visualization"
        msgClick.innerHTML = default_msg;
        accordionHandler(document.getElementById("accordion-MP"), msgClick, default_msg);
        
        msgOut.style.display = "block";

    }
  })
  
}
