h = 1;
w = 0.2;
d = 0.08;
chamf_height = h/10;

CubePoints = [
  [  0,  -d/2,  0 ],  //0
  [ w,  -d/2,  0 ],  //1
  [ w,  d/2,  0 ],  //2
  [  0,  d/2,  0 ],  //3
  [  0,  -d/2,  h ],  //4
  [ w,  -d/2,  h ],  //5
  [ w,  d/2,  h ],  //6
  [  0,  d/2,  h ], //7
  [0,d/2,h+chamf_height],//8
  [w,d/2,h+chamf_height]]; //9
  
CubeFaces = [
  [0,1,2,3],  // bottom
  [4,5,1,0],  // front
  [7,6,5,4],  // top
  [5,6,2,1],  // right
  [6,7,3,2],  // back
  [7,4,0,3], // left
  [8,9,5,4], //big triangle
  [9,6,5],//small triangle face
  [8,4,7],
  [9,8,7,6]];
  
polyhedron( CubePoints, CubeFaces );