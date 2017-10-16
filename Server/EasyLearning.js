var express = require('express');
var app = express();
app.set('view engine', 'pug');
app.set('views', './templates');

var LinearRegression = require('./routes/LinearRegression');
app.use('/LinearRegression', LinearRegression);


var server = app.listen(3000, function () {
  var address = server.address().address;
  var port = server.address().port;
  console.log('Server Running at %s:%s', address, port);
});
