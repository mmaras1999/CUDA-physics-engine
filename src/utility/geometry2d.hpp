// WORK IN PROGRESS
#ifndef GEOMETRY_2D
#define GEOMETRY_2D

#include <cmath>
#include <iostream>

namespace Geometry2d
{
    template <class T>
    struct Vector2d;

    template <class T>
    struct Point 
    {
        T x, y;

        Point() { x = y = 0; }
        Point(T x, T y) : x(x), y(y) {}

        void rotate(T angle)
        {
            auto c = cos(angle);
            auto s = sin(angle);

            auto tmpx = x;
            auto tmpy = y;

            x = tmpx * c - tmpy * s;
            y = tmpy * c + tmpx * s;
        }

        Point<T> operator+ (const Point<T>& rhs) const {
            return Point<T>(x + rhs.x, y + rhs.y);
        }

        Point<T> operator- (const Point<T>& rhs) const {
            return Point<T>(x - rhs.x, y - rhs.y);
        }

        template <class K>
        Point<T> operator* (const K scalar) const {
            return Point<T>(x * scalar, y * scalar);
        }

        template <class K>
        Point<T> operator/ (const K scalar) const {
            return Point<T>(x / scalar, y / scalar);
        }

        Point<T> operator+ (const Vector2d<T>& rhs) const {
            return Point<T>(x + rhs.x, y + rhs.y);
        }

        Point<T> operator- (const Vector2d<T>& rhs) const {
            return Point<T>(x - rhs.x, y - rhs.y);
        }

        operator Vector2d<T>() const {
            return Vector2d<T>(x, y);
        }
    };

    template <class T>
    struct Vector2d 
    {

        T x, y;

        Vector2d() { x = y = 0; }
        Vector2d(T x, T y) : x(x), y(y) {}
        Vector2d(const Point<T>& a, const Point<T>& b) {
            x = b.x - a.x;
            y = b.y - a.y;
        }

        T dot(const Vector2d<T>& rhs) const {
            return x * rhs.x + y * rhs.y;
        }

        T cross (const Vector2d<T>& rhs) const {
            return x * rhs.y - rhs.x * y;
        }

        T norm() const {
            return sqrt(x * x + y * y); 
        }

        Vector2d<T> normalize() const {
            auto n = norm();
            return Vector2d<T>(x / n, y / n);
        }

        Vector2d<T> operator+ (const Vector2d<T>& rhs) const {
            return Vector2d<T>(x + rhs.x, y + rhs.y);
        }

        Vector2d<T> operator- (const Vector2d<T>& rhs) const {
            return Vector2d<T>(x - rhs.x, y - rhs.y);
        }

        template <class K>
        Vector2d<T> operator* (const K scalar) const {
            return Vector2d<T>(x * scalar, y * scalar);
        }

        template <class K>
        Vector2d<T> operator/ (const K scalar) const {
            return Vector2d<T>(x / scalar, y / scalar);
        }
    };

    template <class T>
    struct Line 
    {
        Point<T> p;
        Vector2d<T> dir_vec;
    
        Line(Point<T> point, Vector2d<T> directional_vector)
            : p(point), dir_vec(directional_vector.normalize()) {}
    };

    template <class T>
    struct LineSegment : Line<T> 
    {
        using Line<T>::p;
        using Line<T>::dir_vec;

        Point<T> begin, end;

        LineSegment(Point<T> a, Point<T> b) : Line<T>(a, Vector2d<T>(a, b)), begin(a), end(b) {
        }

        T length() const {
            return Vector2d<T>(begin, end).norm();
        }
    };

    template <class T>
    struct Circle {
        Point<T> center;
        T radius;

        Circle() = default;
        Circle(const Point<T>& p, const T r) : center(p), radius(r) {}
    };

    template <class T>
    struct Triangle 
    {
        Point<T> a, b, c;

        Triangle() {};
        Triangle(const Point<T>& a, const Point<T>& b, const Point<T>& c) 
            : a(a), b(b), c(c) {};

        void rotate(float angle)
        {
            a.rotate(angle);
            b.rotate(angle);
            c.rotate(angle);
        }

        void move(Vector2d<T> vec)
        {
            a = a + vec;
            b = b + vec;
            c = c + vec;
        }
    };

    template <class T>
    bool intersects(const LineSegment<T>& ls1, const LineSegment<T>& ls2)
    {
        auto inter = [](T a, T b, T c, T d)
        {
            if (a > b)
                std::swap(a, b);
            if (c > d)
                std::swap(c, d);
            return std::max(a, c) <= std::min(b, d);
        };

        auto sgn = [](const T& x)
        {
            return (x >= 0) ? x ? 1 : 0 : -1;
        };

        Vector2d<T> a = ls1.begin;
        Vector2d<T> b = ls1.end;
        Vector2d<T> c = ls2.begin;
        Vector2d<T> d = ls2.end;

        if (a.cross(d) == 0 && c.cross(b) == 0)
            return inter(a.x, b.x, c.x, d.x) and inter(a.y, b.y, c.y, d.y);
        
        return sgn((b - a).cross(c - a)) != sgn((b - a).cross(d - a)) &&
            sgn((d - c).cross(a - c)) != sgn((d - c).cross(b - c));
    }

    template <class T>
    bool intersects(const Circle<T>& circle, const LineSegment<T>& ls) 
    {

        Vector2d<T> segment_vec{ls.begin, ls.end};
        Vector2d<T> a_dist_vec{ls.begin, circle.center}; // vector between the first line segment point and the center of the circle
        Vector2d<T> b_dist_vec{ls.end, circle.center};
        T projection_dist = (segment_vec.dot(a_dist_vec) / segment_vec.norm());
        T temp = projection_dist / segment_vec.norm();

        if (temp >= 0.0 and temp <= 1.0) {
            Vector2d<T> z = ls.dir_vec * projection_dist + (Vector2d<T>)(ls.begin);

            Point<T> closest{z.x, z.y};
            
            return Vector2d<T>(closest, circle.center).norm() <= circle.radius;
        }
        else if (temp < 0.0) {
            return a_dist_vec.norm() <= circle.radius;
        }
        else {
            return b_dist_vec.norm() <= circle.radius;
        }
    }

    template <class T>
    bool intersects(const Triangle<T>& tr1, const Triangle<T>& tr2)
    {
        LineSegment<T> tr1_seg[3] = {
            LineSegment<T>(tr1.a, tr1.b),
            LineSegment<T>(tr1.b, tr1.c),
            LineSegment<T>(tr1.c, tr1.a)
        };

        LineSegment<T> tr2_seg[3] = {
            LineSegment<T>(tr2.a, tr2.b),
            LineSegment<T>(tr2.b, tr2.c),
            LineSegment<T>(tr2.c, tr2.a)
        };

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (intersects(tr1_seg[i], tr2_seg[j]))
                    return true;
            }
        }

        return false;
    }
}

#endif
