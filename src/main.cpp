////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Class to store tree
////////////////////////////////////////////////////////////////////////////////
class Node
{
    public:
        AlignedBox3d bbox;
        int parent;   // Index of the parent node (-1 for root)
        int left;     // Index of the left child (-1 for a leaf)
        int right;    // Index of the right child (-1 for a leaf)
        int triangle = -1; 
        int index;
        Matrix3d coordinates;
        bool leaf = false; 
};

class AABBTree
{
    public:
        std::vector<Node> nodes;
        int root;
        AABBTree() = default;                           // Default empty constructor
        AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh
};

////////////////////////////////////////////////////////////////////////////////
// Scene setup, global variables
////////////////////////////////////////////////////////////////////////////////
const std::string data_dir = DATA_DIR;
const std::string filename("raytrace.png");
const std::string mesh_filename(data_dir + "bunny.off");


//Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; //45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 2);

// Triangle Mesh
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)
AABBTree bvh;

//Material for the object, same material for all objects
const Vector4d obj_ambient_color(0.0, 0.5, 0.0, 0);
const Vector4d obj_diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d obj_specular_color(0.2, 0.2, 0.2, 0);
const double obj_specular_exponent = 256.0;
const Vector4d obj_reflection_color(0.7, 0.7, 0.7, 0);

// Precomputed (or otherwise) gradient vectors at each grid node
const int grid_size = 20;
std::vector<std::vector<Vector2d>> grid;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
//Ambient light
const Vector4d ambient_light(0.2, 0.2, 0.2, 0);

//Maximum number of recursive calls
const int max_bounce = 3;

// Objects
std::vector<Vector3d> sphere_centers;
std::vector<double> sphere_radii;
std::vector<Matrix3d> parallelograms;

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //setup tree
    bvh = AABBTree(vertices, facets);

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);




    grid.resize(grid_size + 1);
    for (int i = 0; i < grid_size + 1; ++i)
    {
        grid[i].resize(grid_size + 1);
        for (int j = 0; j < grid_size + 1; ++j)
            grid[i][j] = Vector2d::Random().normalized();
    }

    //Spheres
    sphere_centers.emplace_back(10, 0, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(7, 0.05, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(4, 0.1, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(1, 0.2, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-2, 0.4, 1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-5, 0.8, -1);
    sphere_radii.emplace_back(1);

    sphere_centers.emplace_back(-8, 1.6, 1);
    sphere_radii.emplace_back(1);

    //parallelograms
    parallelograms.emplace_back();
    parallelograms.back() << -100, 100, -100,
        -1.25, 0, -1.2,
        -100, -100, 100;

}

//function use to slice vector, find on stack overflow
template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;
 
    std::vector<T> vec(first, last);
    return vec;
}
 

////////////////////////////////////////////////////////////////////////////////
// BVH Code
////////////////////////////////////////////////////////////////////////////////
AlignedBox3d bbox_from_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c)
{
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}
bool comparison_function(const Vector4d& v1, 
                         const Vector4d& v2) {
  // calculate some comparison_result
  if(v1.x() > v2.x()){
      return true;
  }
  else{
      return false;
  }
}

Node build (const std::vector<Vector4d>& cents, const std::vector<Matrix3d> &verts, const std::vector<AlignedBox3d>& boxes, std::vector<Node*> &nodes, int parent_index=-1){
    if(cents.size() == 1){
        Node leaf;
        leaf.leaf = true;
        int index = cents[0](3);
        leaf.bbox = boxes[index];
        leaf.triangle = index;
        leaf.index = index;
        leaf.parent =  parent_index;
        Matrix3d v = verts[index];
        leaf.coordinates = v;
        nodes.push_back(&leaf);
        int inserted = nodes.size() - 1;
        return leaf;
    }
    int mid = ceil(cents.size()/2);
    std::vector<Vector4d> S1 = slice(cents, 0, mid-1);
    std::vector<Vector4d> S2 = slice(cents, mid, cents.size()-1);
    Node parent;
    AlignedBox3d box;
    parent.bbox = box;
    parent.triangle = -1;
    int size = nodes.size();
    parent.parent = size;
    nodes.push_back(&parent);
    parent_index = nodes.size()-1;
    Node inserted = build(S1, verts, boxes, nodes, parent_index);
    parent.left =  inserted.index;
    Node left = *nodes[inserted.index];
    box.extend(left.bbox);
    Node inserted2 = build(S2, verts, boxes, nodes, parent_index);
    parent.right = inserted2.index;
    Node right = *nodes[inserted2.index];
    box.extend(right.bbox);

    return parent;
}

AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F)
{   
    MatrixXd centroids(F.rows(), V.cols()+1);
    centroids.setZero();
    std::vector<Matrix3d> verts;
    for (int i = 0; i < F.rows(); ++i)
    {   
        Matrix3d v;
        for (int k = 0; k < F.cols(); ++k)
        {
            centroids.row(i) += V.row(F(i, k));
            v.row(k) = V.row(F(i, k));
        }
        verts.push_back(v);
        centroids.row(i) /= F.cols();
        centroids(i, 3) = i;
    }
    std::vector<Vector4d> cents;
    for(int i = 0; i < centroids.rows(); ++i){
        cents.push_back(centroids.row(i));
    }
    std::sort(cents.begin(), cents.end(), comparison_function);
    Vector3d a;
    Vector3d b;
    Vector3d c;
    int index;
    int mid = ceil(cents.size()/2);
    std::vector<AlignedBox3d> boxes;
    for(int i = 0; i < cents.size(); ++i){
        Node node;
        index = cents[i](3);
        a = verts[index].row(0);
        b = verts[index].row(1);
        b = verts[index].row(2);
        AlignedBox3d triangle_box = bbox_from_triangle(a, b, c);
        boxes.push_back(triangle_box);
    }
    std::vector<Node*> nodes; 
    build(cents, verts, boxes, nodes);
    // this->nodes = *nodes;
}

////////////////////////////////////////////////////////////////////////////////
// Intersection code
////////////////////////////////////////////////////////////////////////////////

double ray_triangle_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &a, const Vector3d &b, const Vector3d &c, Vector3d &p, Vector3d &N)
{
    const Vector3d A = a - b;
    const Vector3d B = a - b;
    const Vector3d C = a - c;
    const Vector3d d = ray_direction.normalized(); 
    const Vector3d e = ray_origin;
    const Vector3d w = a - e;
    Matrix3d m;
    m<<A, C, d;
    const Vector3d K = m.colPivHouseholderQr().solve(w);
    const double beta = K[0];
    const double alpha = K[1];
    const double t = K[2];
    const double g = beta + alpha;

    if (t > 0 and beta >= 0 and alpha >= 0 and g <= 1){
        p = e + t*d;
        N = (b-a).cross((c-a)).normalized();
        return t;
    }
    return -1;
}

double ray_sphere_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, int index, Vector3d &p, Vector3d &N)
{
    const Vector3d sphere_center = sphere_centers[index];
    const double sphere_radius = sphere_radii[index];
    double t = -1;
    Vector3d U = ray_direction.normalized();
    Vector3d q = ray_origin - sphere_center;
    double l = 2 * U.dot(q);
    double w = l*l - 4*(q.dot(q) - pow(sphere_radius,2));
    if (w<0)
    {
        return -1;
    }
    else if (w==0){
        t=-l/2;
    }
    else{
        t = (-l-sqrt(w))/2;
    }
    p = ray_origin + t * U;
    N = (-sphere_center+p).normalized();
    return t;
}

//Compute the intersection between a ray and a paralleogram, return -1 if no intersection
double ray_parallelogram_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, int index, Vector3d &p, Vector3d &N)
{
    const Vector3d pgram_origin = parallelograms[index].col(0);
    const Vector3d A = parallelograms[index].col(1);
    const Vector3d B = parallelograms[index].col(2);
    const Vector3d pgram_u = A - pgram_origin;
    const Vector3d pgram_v = B - pgram_origin;
    Matrix3d a;
    Vector3d v;
    a<<pgram_u,pgram_v,-ray_direction;
    v<<ray_origin-pgram_origin;
    Vector3d w = a.colPivHouseholderQr().solve(v);
    if (w(0)<=1 and w(0) > 0 and w(1) <=1 and w(1)>0)
    {
        p = pgram_origin + w(0)*pgram_u+w(1)*pgram_v;
        N = -pgram_u.cross(pgram_v).normalized();
        return w(2);
    } 
    
    return -1;
}

bool ray_box_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, const AlignedBox3d &box)
{
    const Vector3d ray_direct =  ray_direction.normalized();
    Vector3d max = box.max();
    Vector3d min = box.min();
    const double width = max[0] - min[0];
    const double length = max[1] - min[1];
    const double height = max[2] - min[2];
    const double a = 1 / ray_direction.x();
    const double b = 1 / ray_direction.y();
    const double c = 1/ ray_direction.z();
    double t_xmin;
    double t_xmax;
    double t_ymin;
    double t_ymax;
    double t_zmin;
    double t_zmax;
    if(ray_direction.x() > 0){
        t_xmin =  a*(min.x() - ray_origin.x());
        t_xmax = a*(max.x() - ray_origin.x());
    }else{
        t_xmax =  a*(min.x() - ray_origin.x());
        t_xmin = a*(max.x() - ray_origin.x());
    }
    if(ray_direction.y() > 0){
        t_ymin = b*(min.y() - ray_origin.y()) ;
        t_ymax = b*(max.y() - ray_origin.y());
    }else{
        t_ymax = b*(min.y() - ray_origin.y());
        t_ymin = b*(max.y() - ray_origin.y());
    }
    if(ray_direction.z() > 0){
        t_zmin = c*(min.z() - ray_origin.z());
        t_zmax = c*(max.z() - ray_origin.z());
    }else{
        t_zmax= c*(min.z() - ray_origin.z());
        t_zmin= c*(max.z() - ray_origin.z());
    }
    if (t_xmin > t_xmax or t_ymin > t_ymax or t_zmin > t_zmax){
        return false;
    }else{
        return true;
    }
    return false;
}

//return closet t
int bvh_search(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N, Vector3d& tmp_p, Vector3d& tmp_N){
    int closest_index = -1;
    double closest_t = std::numeric_limits<double>::max();
    std::vector<Node> nodes = bvh.nodes;
    Node root = nodes[0];
    bool inter = ray_box_intersection(ray_direction, ray_direction, nodes[0].bbox);
    Node curr = nodes[0];
    int i = 0;
    while(inter){
        Node left = nodes[curr.left];
        Node right = nodes[curr.right];
        bool inter_left = ray_box_intersection(ray_direction, ray_direction, left.bbox);
        bool inter_right = ray_box_intersection(ray_direction, ray_direction, right.bbox);
        if(inter_left == true){
            if(left.triangle != -1){
                Matrix3d v = left.coordinates;
                const double t = ray_triangle_intersection(ray_origin,ray_direction, v.row(0), v.row(1), v.row(2), tmp_p, tmp_N);
                if(t>0){
                    if (t < closest_t){
                        closest_index = i;
                        closest_t = t;
                        p = tmp_p;
                        N = tmp_N;
                    }
                }
            }else{
                curr = left;
                continue;
            }

        }
        if(inter_right == true){
            curr = right;
            continue;
        }else{
            break;
        }
        i++;

    }
    return closest_index;
}

int find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N)
{
    Vector3d tmp_p, tmp_N;
    int closest_index = -1;
    double closest_t = std::numeric_limits<double>::max();
    for(int i=0; i < facets.rows(); ++i){
        const Vector3d a (vertices(facets(i,0), 0),vertices(facets(i,0), 1),vertices(facets(i,0), 2));
        const Vector3d b (vertices(facets(i,1), 0),vertices(facets(i,1), 1),vertices(facets(i,1), 2));
        const Vector3d c (vertices(facets(i,2), 0),vertices(facets(i,2), 1),vertices(facets(i,2), 2));
        const double t = ray_triangle_intersection(ray_origin, ray_direction, a, b, c, tmp_p, tmp_N);
        if(t>0){
            if (t < closest_t){
                closest_index = i;
                closest_t = t;
                p = tmp_p;
                N = tmp_N;
            }
        }
    }
    for (int i = 0; i < sphere_centers.size(); ++i)
    {
        //returns t and writes on tmp_p and tmp_N
        const double t = ray_sphere_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
        //We have intersection
        if (t >= 0)
        {
            if (t < closest_t)
            {
                closest_index = facets.rows() + i;
                closest_t = t;
                p = tmp_p;
                N = tmp_N;
            }
        }
    }

    for (int i = 0; i < parallelograms.size(); ++i)
    {
        const double t = ray_parallelogram_intersection(ray_origin, ray_direction, i, tmp_p, tmp_N);
        if (t >= 0)
        {
            if (t < closest_t)
            {
                closest_index = facets.rows() + sphere_centers.size() + i;
                closest_t = t;
                p = tmp_p;
                N = tmp_N;
            }
        }
    }
    return closest_index;
}

bool is_light_visible(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &light_position)
{
    Vector3d p, N;
    const int intersection = find_nearest_object(ray_origin, ray_direction, p, N);
    //no intersection between shadow ray and object
    if (intersection == -1){
        return false;
    }
    //has object in between
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Raytracer code
////////////////////////////////////////////////////////////////////////////////

Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int max_bounce)
{
    //Intersection point and normal, these are output of find_nearest_object
    Vector3d p, N;
    const int nearest_object = find_nearest_object(ray_origin, ray_direction, p, N);

    if (nearest_object < 0)
    {
        // Return a transparent color
        return Vector4d(0, 0, 0, 0);
    }
    // Ambient light contribution
    const Vector4d ambient_color = obj_ambient_color.array() * ambient_light.array();
    // Punctual lights contribution (direct lighting)
    Vector4d lights_color(0, 0, 0, 0);
    for (int i = 0; i < light_positions.size(); ++i)
    {
        const Vector3d &light_position = light_positions[i];
        const Vector4d &light_color = light_colors[i];

        Vector4d diff_color = obj_diffuse_color;
        const Vector3d Li = (light_position - p).normalized();
        // TODO: Add shading parameters
        const int visible = is_light_visible(p+0.0001*Li, Li, light_position);
        if (visible == true){
            continue;
        }
 
        // Diffuse contribution
        const Vector4d diffuse = diff_color * std::max(Li.dot(N), 0.0);

        // Specular contribution
        const Vector3d Hi = (Li - ray_direction).normalized();
        const Vector4d specular = obj_specular_color * std::pow(std::max(N.dot(Hi), 0.0), obj_specular_exponent);
        // Vector3d specular(0, 0, 0);

        // Attenuate lights according to the squared distance to the lights
        const Vector3d D = light_position - p;
        lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
    }

    Vector4d refl_color = obj_reflection_color;
    if (nearest_object == 4)
    {
        refl_color = Vector4d(0.5, 0.5, 0.5, 0);
    }
    Vector4d reflection_color(0, 0, 0, 0);
    Vector3d d = ray_direction.normalized();
    Vector3d r = d - 2 * (d.dot(N)* N);
    reflection_color = refl_color.cwiseProduct(shoot_ray(p + 0.0001*r, r, max_bounce-1));
    Vector4d refraction_color(0, 0, 0, 0);
    Vector4d C = ambient_color + lights_color + reflection_color + refraction_color;
    C(3) = 1;

    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()
{
    std::cout << "Simple ray tracer." << std::endl;

    int w = 640;
    int h = 480;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask
    double aspect_ratio = double(w) / double(h);
    double image_y = focal_length * tan(field_of_view/2); //TODO: compute the correct pixels size
    double image_x = image_y * aspect_ratio; //TODO: compute the correct pixels size
    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    const Vector3d image_origin(-image_x, image_y, camera_position[2] - focal_length);
    const Vector3d x_displacement(2.0 / w * image_x, 0, 0);
    const Vector3d y_displacement(0, -2.0 / h * image_y, 0);

    for (unsigned i = 0; i < w; ++i)
    {
        for (unsigned j = 0; j < h; ++j)
        {
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            // Prepare the ray
            Vector3d ray_origin;
            Vector3d ray_direction;

            if (is_perspective)
            {
                // Perspective camera
                ray_origin = camera_position;
                ray_direction = (pixel_center - camera_position).normalized();
            }
            else
            {
                // Orthographic camera
                ray_origin = pixel_center;
                ray_direction = Vector3d(0, 0, -1);
            }

            const Vector4d C = shoot_ray(ray_origin, ray_direction, max_bounce);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = C(3);
        }
    }
    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    setup_scene();
    raytrace_scene();
    return 0;
}
