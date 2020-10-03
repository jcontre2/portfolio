$(document).ready(function(){
  var nickName = ""
  $("#click").click(function(){
    $.get("/id",function(response){
      window.location = "/chatroom/" + response.roomId;
      console.log(response.roomId+" Home");
    });
  });
});
