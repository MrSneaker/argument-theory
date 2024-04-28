'use strict';

window.addEventListener('load', function () {

  console.log("Hello World!");

  $(document).ready(function(){
    $("#ruleWeightsCheck").change(function(){
      if(this.checked){
        $("#preferencesTextarea").prop("disabled", true);
      } else {
        $("#preferencesTextarea").prop("disabled", false);
      }
    });
  });

});