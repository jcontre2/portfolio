//all the modulos are installes
var http = require('http');
var express = require('express');
var bodyParser = require('body-parser');
var anyDB = require('any-db-mysql');
var engines = require('consolidate');
var app = express();
var server = http.createServer(app);

var io = require('socket.io').listen(server);
//telling app to use body-parser
app.use(bodyParser.urlencoded({extended:false}));
app.use(bodyParser.json());
app.use(express.static(__dirname + '/templates'));
app.use(express.static(__dirname + '/scripts'));

var conn = anyDB.createConnection('sqlite3://chatroom.db');

conn.query('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT,nickname TEXT,body TEXT,time INTEGER)')
.on('close', function(data){
  console.log("db Success");
}).on('error', function(){
  console.log("db Error");
});

conn.query('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT,nickname TEXT)')
.on('close', function(data){
  console.log("users Success");
}).on('error', function(){
  console.log("users Error");
});
//adjusting to use hogan
app.engine('html', engines.hogan);
app.set('views', __dirname + '/templates');
app.set('view engine', 'html');
app.use(express.static(__dirname + '/templates'));
app.use(express.static(__dirname + '/scripts'));

//generating Identifier
function generateId() {
  var chars = 'ABCDEFGHJKLMNOPQRSTUVWXYZ23456789';

  var roomId = '';
  for (var i = 0; i<6; i++){
    roomId += chars.charAt(Math.floor(Math.random()*chars.length));
  }
  return roomId;
};

app.get('/', function(request, response){
  console.log("~Request received: ", request.method, request.url);
  response.render('home.html');
});

app.get('/id', function(request, response){
  var id = generateId();
  response.json({roomId: id});
});

app.get('/:roomName', function(request,response){
  response.render('room.html',{roomName:request.params.roomName});
});

io.sockets.on('connection', function(socket){
  socket.on('join',function(roomName,nickname,callback){
    socket.join(roomName);
    socket.roomName = roomName;
    socket.nickname = nickname;
    var sql1 = 'INSERT INTO users (room,nickname) VALUES ($1,$2)';
    var q = conn.query(sql1,[roomName,nickname],function(error, result){
      if (error!=null){
        console.log(error);
      }
    });
    var sql2= 'SELECT id, nickname, body FROM messages WHERE room=$1 ORDER BY id ASC';
    var q = conn.query(sql2, [roomName], function(error, result){
      var messages= result.rows;
      callback(messages);
    });

    var sql3= 'SELECT id, nickname FROM users WHERE room=$1 ORDER BY id ASC';
    var q = conn.query(sql3, [roomName], function(error, result){
      var members= result.rows;
      io.sockets.in(roomName).emit('members',members);
    });
    io.sockets.in(roomName).emit('newMember', nickname);
  });

  socket.on('messageTo',function(nickname,message){
    var roomName = Object.keys(io.sockets.adapter.sids[socket.id])[1];
    var sql = 'INSERT INTO messages (room,nickname,body,time) VALUES ($1,$2,$3,$4)';
    var q = conn.query(sql,[roomName,nickname,message,0],function(error, result){
      if (error!=null){
        console.log(error);
      }
    });
    io.sockets.in(roomName).emit('messageFrom', nickname, message);
  });

  socket.on('disconnect',function(){
    var roomName = socket.roomName;
    var disMember = socket.nickname;
    io.sockets.in(roomName).emit('memberLeft', disMember);
    var sql = 'DELETE FROM users WHERE nickname=$1 AND room=$2';
    var q = conn.query(sql,[disMember,roomName],function(error, result){

    });
    var sql1 = 'SELECT nickname FROM users WHERE room=$1';
    var q = conn.query(sql1,[roomName],function(error, result){
      var members=result.rows;
      io.sockets.in(roomName).emit('members',members);
    });
  });

  socket.on('nameChange',function(old, nombre){
    var sql = 'DELETE FROM users WHERE nickname=$1 AND room=$2';
    var q = conn.query(sql,[old,socket.roomName]);
    var sql1 = 'INSERT INTO users (room,nickname) VALUES ($1,$2)';
    var q = conn.query(sql1,[socket.roomName,nombre],function(error, result){

    });
    var sql3= 'SELECT id, nickname FROM users WHERE room=$1 ORDER BY id ASC';
    var q = conn.query(sql3, [socket.roomName], function(error, result){
      var members= result.rows;
      io.sockets.in(socket.roomName).emit('members',members);
    });
    io.sockets.in(socket.roomName).emit('nameChange',old,nombre);
  });

  socket.on('error',function(){
    console.log('error when closing the connection');
  });
});

server.listen(80, function(){
  console.log("server is on port 80");
});
