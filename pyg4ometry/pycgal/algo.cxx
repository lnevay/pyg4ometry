#include "algo.h"

#include <sstream>

/*********************************************
Polyhedron
*********************************************/
Polyhedron3::Polyhedron3() {}
Polyhedron3::~Polyhedron3() {}

/*********************************************
Nef Polyhedron
*********************************************/
NefPolyhedron3::NefPolyhedron3() {}
NefPolyhedron3::~NefPolyhedron3() {}

/*********************************************
Surface Mesh
*********************************************/
SurfaceMesh::SurfaceMesh() {
  _surfacemesh = new Surface_mesh();
}

SurfaceMesh::SurfaceMesh(const SurfaceMesh &mesh) {
  _surfacemesh = new Surface_mesh(*(mesh._surfacemesh));
}

SurfaceMesh::SurfaceMesh(Surface_mesh *mesh) {
  _surfacemesh = mesh;
}

SurfaceMesh::SurfaceMesh(py::list vertices, 
			 py::list faces)    {
  _surfacemesh = new Surface_mesh();
  
  // loop over vertices 
  for (py::handle vHandle : vertices) {
    py::list vList = vHandle.cast<py::list>();
    add_vertex(vList[0].cast<double>(),
	       vList[1].cast<double>(),
	       vList[2].cast<double>());
  }
  
  // loop over faces 
  for (py::handle fHandle : faces) {
    py::list fList = fHandle.cast<py::list>();
    add_face(fList[0].cast<int>(),
	     fList[1].cast<int>(),
	     fList[2].cast<int>());	    
  }
}

SurfaceMesh::SurfaceMesh(py::array_t<double> vertices,
			 py::array_t<int>    faces)    {
  _surfacemesh = new Surface_mesh();

  // loop over vertices 

  // loop over faces 
}

SurfaceMesh::~SurfaceMesh() {
  delete _surfacemesh;
}

std::size_t SurfaceMesh::add_vertex(double x, double y, double z) {
  return _surfacemesh->add_vertex(Point(x,y,z));
}

std::size_t SurfaceMesh::add_face(std::size_t i, std::size_t j, std::size_t k) {
  return _surfacemesh->add_face(Surface_mesh::Vertex_index(i),
				Surface_mesh::Vertex_index(j),
				Surface_mesh::Vertex_index(k));
}

std::size_t SurfaceMesh::add_face(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {

  _surfacemesh->add_face(Surface_mesh::Vertex_index(i),
			 Surface_mesh::Vertex_index(j),
			 Surface_mesh::Vertex_index(k));

  _surfacemesh->add_face(Surface_mesh::Vertex_index(i),
			 Surface_mesh::Vertex_index(k),
			 Surface_mesh::Vertex_index(l));
  /*
  return _surfacemesh->add_face(Surface_mesh::Vertex_index(i),
				Surface_mesh::Vertex_index(j),
				Surface_mesh::Vertex_index(k),
				Surface_mesh::Vertex_index(l));
  */
}

void SurfaceMesh::translate(double x, double y, double z) {
  Aff_transformation_3 transl(CGAL::TRANSLATION, Vector_3(x, y, z));  
  CGAL::Polygon_mesh_processing::transform(transl,*_surfacemesh);
}

void SurfaceMesh::transform(double m11, double m12, double m13,
			    double m21, double m22, double m23,
			    double m31, double m32, double m33) {
  Aff_transformation_3 tform(m11,m12,m13,
			     m21,m22,m23,
			     m31,m32,m33,1);  
  CGAL::Polygon_mesh_processing::transform(tform,*_surfacemesh);
}


SurfaceMesh* SurfaceMesh::unioN(SurfaceMesh &mesh2) {
  Surface_mesh *out = new Surface_mesh();
  bool valid_union = CGAL::Polygon_mesh_processing::corefine_and_compute_union(*_surfacemesh,*(mesh2._surfacemesh), *out);
  return new SurfaceMesh(out);
}

SurfaceMesh* SurfaceMesh::intersect(SurfaceMesh &mesh2) {
  Surface_mesh *out = new Surface_mesh();
  bool valid_intersection = CGAL::Polygon_mesh_processing::corefine_and_compute_intersection(*_surfacemesh,*(mesh2._surfacemesh), *out);
  return new SurfaceMesh(out);
}

SurfaceMesh* SurfaceMesh::subtract(SurfaceMesh &mesh2) {
  Surface_mesh *out = new Surface_mesh();
  bool valid_difference = CGAL::Polygon_mesh_processing::corefine_and_compute_difference(*_surfacemesh,*(mesh2._surfacemesh), *out);
  return new SurfaceMesh(out);
}

py::list* SurfaceMesh::toVerticesAndPolygons() {
  py::list *verts = new py::list();
  py::list *polys = new py::list();

  //std::vector<std::vector<double>> verts;
  //std::vector<std::vector<int>>    polys;
  
  Surface_mesh::Point p;
  for(Surface_mesh::Vertex_index vd : _surfacemesh->vertices()) {
    py::list *v = new py::list();
    p = _surfacemesh->point(vd);
    v->append(CGAL::to_double(p.x()));
    v->append(CGAL::to_double(p.y()));
    v->append(CGAL::to_double(p.z()));    
    verts->append(v);
  }

  int iCount = 0;
  for(Surface_mesh::Face_index fd : _surfacemesh->faces()) {
    py::list *p = new py::list();
    for(Surface_mesh::Halfedge_index hd :  CGAL::halfedges_around_face(_surfacemesh->halfedge(fd),*_surfacemesh)) {
      p->append((int)_surfacemesh->source(hd));
    }
    polys->append(p);
    ++iCount;
  }

  py::list *ret = new py::list();
  ret->append(verts);
  ret->append(polys);
  ret->append(iCount);
  
  return ret;
}

int SurfaceMesh::number_of_faces() {
  return _surfacemesh->number_of_faces();
}

std::string SurfaceMesh::toString() { 
  std::ostringstream sstream;
  sstream << *_surfacemesh;
  return sstream.str();
}

/*********************************************
Polygon2
*********************************************/
Polygon2::Polygon2() {
  _polygon = new Polygon_2();
};

Polygon2::Polygon2(const Polygon2 &poly) {
  _polygon = new Polygon_2(*(poly._polygon));
}

Polygon2::Polygon2(const Polygon_2 &poly) {
  _polygon = new Polygon_2(poly);
}

Polygon2::Polygon2(py::list &p) {
  _polygon = new Polygon_2();
  for (py::handle pHandle : p) {
    py::list pList = pHandle.cast<py::list>();
    push_back(pList[0].cast<double>(), pList[1].cast<double>());
  }
}

Polygon2::Polygon2(py::list &x, py::list &y) {
  _polygon = new Polygon_2();

  int i = 0; 
  for (py::handle pHandle : x) {
    push_back(pHandle.cast<double>(), y[i].cast<double>());
    i++;
  }
}

Polygon2::Polygon2(py::array_t<double> &array) {
  _polygon = new Polygon_2();

  py::buffer_info buf = array.request();
  
  if (buf.ndim != 2)
    throw std::runtime_error("numpy.ndarray dims must be 2");
  
  double *ptr = (double*)buf.ptr;
  int n1 = buf.shape[0];
  int n2 = buf.shape[1];

  for(int i = 0; i< n1; i++) {
    push_back(ptr[i*n2+0],ptr[i*n2+1]);
  }
}

Polygon2::Polygon2(py::array_t<double> &array_x, 
		   py::array_t<double> &array_y) {
  _polygon = new Polygon_2();

  py::buffer_info buf_x = array_x.request();
  py::buffer_info buf_y = array_y.request();

  if (buf_x.shape[0] != buf_y.shape[0])
    throw std::runtime_error("numpy.ndarray arrays need to be same length");

  int n = buf_x.shape[0];

  double *ptr_x = (double*)buf_x.ptr;
  double *ptr_y = (double*)buf_y.ptr;

  for(int i = 0; i< n; i++) {
    push_back(ptr_x[i],ptr_y[i]);
  }
}

Polygon2::~Polygon2() {
  delete _polygon;
};

void Polygon2::push_back(double x , double y) {
  _polygon->push_back(Point_2(x,y));
}

void Polygon2::push_back(py::list &list) {
  double x = list[0].cast<double>();
  double y = list[1].cast<double>();
  push_back(x,y);
}

void Polygon2::push_back(py::array_t<double> &array) { 
  py::buffer_info buf = array.request();
  double* ptr = (double*)buf.ptr;
  double x = ptr[0];
  double y = ptr[1];  
  push_back(x,y);
}

std::size_t Polygon2::size() const {
  return _polygon->size();
}

std::string Polygon2::toString() const {
  std::ostringstream sstream;
  sstream << "Polygon2[" << std::endl;
  for(auto vi = _polygon->vertices_begin(); vi != _polygon->vertices_end(); ++vi) {
    sstream << *vi << std::endl;  
  }
  sstream << "]";
  return sstream.str();
}

void Polygon2::clear() {
  _polygon->clear();
}

void Polygon2::reverse_orientation() {
  _polygon->reverse_orientation();
}


bool Polygon2::is_simple() const {
  return _polygon->is_simple();
}

bool Polygon2::is_convex() const {
  return _polygon->is_convex();
}

int Polygon2::orientation() const {
  return _polygon->orientation();
}

std::list<Polygon2> Polygon2::optimal_convex_partition() {
  std::list<Polygon_2> partition_polys_temp;
  std::list<Polygon2>  partition_polys;

  // CW or CCW
  if(orientation() != 1) {
    reverse_orientation();
  }

  CGAL::optimal_convex_partition_2(_polygon->vertices_begin(),
				   _polygon->vertices_end(),
				   std::back_inserter(partition_polys_temp));

  for(auto p : partition_polys_temp) {
    partition_polys.push_back(Polygon2(p));
  }
 
  return partition_polys;
}

/*********************************************
PYBIND
*********************************************/
PYBIND11_MODULE(algo, m) {

  py::class_<Polyhedron3>(m,"Polyhedron3")
    .def(py::init<>());

  py::class_<NefPolyhedron3>(m,"NefPolyhedron3")
    .def(py::init<>());

  py::class_<SurfaceMesh>(m,"SurfaceMesh")
    .def(py::init<>())
    .def(py::init<py::list &, py::list &>())
    .def("add_vertex",&SurfaceMesh::add_vertex)
    .def("add_face",(std::size_t (SurfaceMesh::*)(std::size_t, std::size_t, std::size_t)) &SurfaceMesh::add_face)
    .def("add_face",(std::size_t (SurfaceMesh::*)(std::size_t, std::size_t, std::size_t, std::size_t)) &SurfaceMesh::add_face)
    .def("translate",&SurfaceMesh::translate)
    .def("transform",&SurfaceMesh::transform)
    .def("union",&SurfaceMesh::unioN)
    .def("intersect",&SurfaceMesh::intersect)
    .def("subtract",&SurfaceMesh::subtract)
    .def("toVerticesAndPolygons",&SurfaceMesh::toVerticesAndPolygons)
    .def("number_of_faces",&SurfaceMesh::number_of_faces)
    .def("__repr__",&SurfaceMesh::toString);

  py::class_<Polygon2>(m,"Polygon2")
    .def(py::init<>())
    .def(py::init<py::list &>())
    .def(py::init<py::list &, py::list &>())
    .def(py::init<py::array_t<double> &>())
    .def(py::init<py::array_t<double> &, py::array_t<double> &>())
    .def("push_back",(void (Polygon2::*)(double, double)) &Polygon2::push_back)
    .def("push_back",(void (Polygon2::*)(py::list &)) &Polygon2::push_back)
    .def("push_back",(void (Polygon2::*)(py::array_t<double> &)) &Polygon2::push_back)
    .def("__repr__",&Polygon2::toString)
    .def("__len__",&Polygon2::size)
    .def("clear",&Polygon2::clear)
    .def("reverse_orientation",&Polygon2::reverse_orientation)
    .def("is_simple",&Polygon2::is_simple)
    .def("is_convex",&Polygon2::is_convex)
    .def("orientation",&Polygon2::orientation)
    .def("optimal_convex_partition",&Polygon2::optimal_convex_partition);
}
