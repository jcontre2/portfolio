
var nombre = null;
var socket = null;
$(document).ready(function(){
  socket = io.connect();
  nombre = window.prompt('What would you like your nickname to be?');

  function meta(name) {
      var tag = document.querySelector('meta[name=' + name + ']');
      if (tag != null)
          return tag.content;
      return '';
  }

  var roomName = meta('roomName');
  var messageForm =$('#messageForm').submit(sendMessage);

  function sendMessage(){
    event.preventDefault();
    var message = $('#messageField').val();
    socket.emit('messageTo',nombre,message);
  }

  socket.on('messageFrom',function(nickName,message){
    var ul = $('#list');
    var li = $('<li></li>');
    li.html('<strong>'+nickName+'</strong>'+'<br>'+message);
    ul.append(li);

  });

    // join the room
  socket.emit('join', meta('roomName'), nombre, function(messages){
    var ul = $('#list');
    ul.empty();
    for (var i = 0; i < messages.length; i++){
      var nickname = messages[i]["nickname"]
      var body = messages[i]["body"]
      var li = $('<li></li>');
      li.html('<strong>'+nickname+'</strong>'+'<br>'+body);
      ul.append(li);
    }
    var lit = $('<li></li>');
    lit.html('<strong>'+nombre+' has joined the room</strong>');
    ul.append(lit);
  });

  socket.on('members', function(members){
    var ul = $('#members');
    ul.empty();
    var li = $('<li><strong>Members in room</strong></li>');
    ul.append(li);
    for (var i = 0; i < members.length; i++){
      var nickname = members[i]["nickname"]
      var li = $('<li></li>');
      li.html(nickname);
      ul.append(li);
    }
  });

  socket.on('newMember', function(newMember){
    var ul = $('#list');
    var li = $('<li></li>');
    li.html('<strong>'+newMember+' has joined the room</strong>'+'<br>');
    ul.append(li);
  });

  socket.on('nameChange',function(old,newName){
    var ul = $('#list');
    var lit = $('<li></li>');
    lit.html('<strong>'+old+' has changed name to '+newName+'</strong>');
    ul.append(lit);
  });

  socket.on('memberLeft', function(disMember){
    var ul = $('#list');
    var li = $('<li></li>');
    li.html('<strong>'+disMember+' has left the room</strong>'+'<br>');
    ul.append(li);
  });
});
function changeNickname(){
  var old = nombre;
  nombre = window.prompt('What would you like your new nickname to be?');
  socket.emit('nameChange',old, nombre);
}
