#! /usr/bin/env python3

# coding: utf-8
#
# Copyright (C) 2020 Matthias Uhl <uhl@theo2.physik.uni-stuttgart.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import unittest as ut
import warnings
from StepPotential import *


def testdiff(poly,expr):
    return sym.simplify(poly.as_expr()-expr)



class TestPolyExp(ut.TestCase):
    def setUp(self):
        self.x = x
        self.constPoly = PolyExp([1],-3,1,self.x)
        self.quadPoly = PolyExp([4,-3,1],5,2,self.x)
        self.longPoly = PolyExp([15,4,-3,8,7,-4,20],5,1,self.x)
        warnings.simplefilter("error")
        
    def test_construct_PolyExp_from_list(self):
        t = PolyExp([2,5,6,7],3,1,self.x)
        
    def test_construct_PolyExp_from_Poly(self):
        x = self.x
        p = sym.Poly(2*x**3 + 5*x**2 + 6*x +7, x )
        t = PolyExp(p,3,1,self.x)
    
    def test_construct_PolyExp_from_expression(self):
        x = self.x
        p = 2*x**3 + 5*x**2 + 6*x +7
        t = PolyExp(p,3,1,self.x)
        
    def test_construct_PolyExp_from_string(self):
        p = "2*x**3 + 5*x**2 + 6*x + 7"
        t = PolyExp(p,3,1,self.x)
    
    def test_expression_conversion(self):
        x = self.x
        diff = sym.simplify(self.constPoly.as_expr() - sym.exp(-3*x))
        self.assertEqual(diff,0)
        
        diff = sym.simplify(self.quadPoly.as_expr() - 2*sym.exp(5*x)*(4*x**2 - 3*x +1))
        self.assertEqual(diff,0)
        
        diff = sym.simplify(self.longPoly.as_expr() - sym.exp(5*x)*(15*x**6 + 4*x**5 
                                                                    -3*x**4 +8*x**3 +7*x**2 
                                                                    - 4*x +20))
        self.assertEqual(diff,0)
        
    def test_multiplication(self):
        x = self.x
        diff = testdiff(5*self.quadPoly, 10*sym.exp(5*x)*(4*x**2 - 3*x +1))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.quadPoly*5, 10*sym.exp(5*x)*(4*x**2 - 3*x +1))
        self.assertEqual(diff,0)
        
        diff = testdiff(sym.exp(5)*self.quadPoly, 2*sym.exp(5*x+5)*(4*x**2 - 3*x +1))
        self.assertEqual(diff,0)
        
    def test_exponential_multiplication(self):
        x = self.x
        
        diff = testdiff(self.constPoly.exp_mul(-3), sym.exp(-6*x))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.quadPoly.exp_mul(5), 2*(4*x*x-3*x+1)*sym.exp(10*x))
        self.assertEqual(diff,0)
        
        
    def test_division(self):
        x = self.x
        diff = testdiff(self.quadPoly/2, sym.exp(5*x)*(4*x**2 - 3*x +1))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.quadPoly/sym.exp(-5),2*sym.exp(5*x+5)*(4*x**2 - 3*x +1))
        self.assertEqual(diff,0)
        
    def test_mult_div_with_non_constant(self):
        with self.assertRaises(ValueError):
            self.quadPoly*sym.exp(5*self.x)
            
        with self.assertRaises(ValueError):
            self.quadPoly/sym.exp(5*self.x)
            
    def test_addition(self):
        x = self.x
        s = self.quadPoly + self.longPoly
        diff = testdiff(s,sym.exp(5*x)*(15*x**6 + 4*x**5 -3*x**4 +8*x**3 +15*x**2 - 10*x +22))
        self.assertEqual(diff,0)
        
        with self.assertRaises(ValueError):
            self.quadPoly + self.constPoly
            
    def test_subtraction(self):
        x = self.x
        d = self.longPoly-self.quadPoly
        diff = testdiff(d,sym.exp(5*x)*(15*x**6 + 4*x**5 -3*x**4 +8*x**3 -x**2 +2*x +18))
        self.assertEqual(diff,0)
                        
            
    def test_exp_integrate(self):
        x = self.x
        
        diff = testdiff(self.constPoly.exp_integrate(3),x)
        self.assertEqual(diff,0)
        
        diff = testdiff(self.constPoly.exp_integrate(-2),-sym.exp(-5*x)/5)
        self.assertEqual(diff,0)
        
        diff = testdiff(self.quadPoly.exp_integrate(-5),8*x**3/3 - 6*x**2/2 + 2*x)
        self.assertEqual(diff,0)
        
        diff = testdiff(self.quadPoly.exp_integrate(3),sym.exp(8*x)*(3/8 - x + x**2))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.longPoly.exp_integrate(-5),15*x**7/7 +  2*x**6/3 -3*x**5/5 +2*x**4 +7*x**3/3 -2*x**2 +20*x)
        self.assertEqual(diff,0)
        
        diff = testdiff(self.longPoly.exp_integrate(-6),sym.exp(-x)*(-15*x**6 -94*x**5 -467*x**4 
                                                                    -1876*x**3 -5635*x**2 -11266*x
                                                                    -11286))
        self.assertEqual(diff,0)
        

    def test_gauss_integrate(self):
        x = self.x
        diff = sym.simplify(self.constPoly.gauss_integrate() 
                            - sym.exp(sym.Rational(9,4))*sym.sqrt(sym.pi)*sym.erf(x+sym.Rational(3,2))/2)
        self.assertEqual(diff,0)
        
        diff = sym.simplify(self.quadPoly.gauss_integrate() 
                           -(-sym.exp(-(-5+x)*x)*(7+4*x) 
                             - 41*sym.exp(sym.Rational(25,4))*sym.sqrt(sym.pi)*sym.erf(sym.Rational(5,2)-x)/2))
        self.assertEqual(diff,0)
        
        diff = sym.simplify(self.longPoly.gauss_integrate()
                           -sym.Rational(1,128)*(-2*sym.exp(-(-5+x)*x)*(118319+41790*x+
                                                13972*x*x+4424*x**3+1328*x**4+480*x**5) 
                                    - 634665*sym.exp(sym.Rational(25,4))*sym.sqrt(sym.pi)*sym.erf(sym.Rational(5,2)-x)))
        self.assertEqual(diff,0)
    
    def test_integration_of_zero(self):
        x = self.x
        
        t = PolyExp(sym.Poly(0,x),0)
        diff = testdiff(t,0)
        self.assertEqual(diff,0)
        
    def test_xscale(self):
        x = self.x
        
        diff = testdiff(self.constPoly.xscale(-2), sym.exp(6*x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.quadPoly.xscale(5), 2*sym.exp(25*x)*(1-15*x+100*x*x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.longPoly.xscale(2), sym.exp(10*x)*(20-8*x+28*x*x 
                                                +64*x**3 -48*x**4+128*x**5+960*x**6))
        self.assertEqual(diff,0)
        
        
    def test_shift(self):
        x = self.x
        pnew = self.constPoly.shift(7)
        diff = testdiff(pnew, sym.exp(-3*x+21))
        self.assertEqual(diff,0)
        
        pnew = self.quadPoly.shift(-3)
        diff = testdiff(pnew, 2*sym.exp(5*(3+x))*(28+21*x+4*x**2))
        self.assertEqual(diff,0)
        
        pnew  = self.longPoly.shift(1)
        diff = testdiff(pnew, sym.exp(-5+5*x)*(31 -52*x +150*x**2 -240*x**3 + 202*x**4 -86*x**5 +15*x**6))
        self.assertEqual(diff,0)
        
    def test_mirror(self):
        x = self.x
        
        pnew = self.constPoly.mirror(3)
        diff = testdiff(pnew, sym.exp(-3*(6-x)))
        self.assertEqual(diff,0)
        
        pnew = self.quadPoly.mirror(-5)
        diff = testdiff(pnew, 2*sym.exp(-5*(10+x))*(431 +83*x+4*x*x))
        self.assertEqual(diff,0)
        
        pnew = self.longPoly.mirror(20)
        diff = testdiff(pnew, sym.exp(-5*(-40+x))*(61842443060 - 9266470956*x +578532167*x**2
                                                  -19263528*x**3 + 360797*x**4 -3604*x**5 +15*x**6))
        self.assertEqual(diff,0)
        
    def test_shift_mirror(self):
        x = self.x
        pnew = self.constPoly.shift_mirror(2,1)
        ptest = self.constPoly.shift(-1).mirror(1)
        diff = testdiff(pnew, ptest.as_expr())
        self.assertEqual(diff,0)
        
        xl = sym.floor((random.random()*20-10)*10000)/10000
        xr = sym.floor((random.random()*20-10)*10000)/10000
        
        pnew = self.longPoly.shift_mirror(xl,xr)
        ptest = self.longPoly.shift(xr-xl).mirror(xr)
        diff = testdiff(pnew, ptest.as_expr())
        self.assertEqual(diff,0)
        
        
        
    def test_copy(self):
        poly1 = self.longPoly.shift(4)
        poly2 = self.longPoly.shift(4)
        diff = testdiff(poly1, poly2.as_expr())
        self.assertEqual(diff,0)
        
    def test_linear_poly(self): 
        """Linear polynomials behave in a odd way because expand does nothing when used on them.
        However, we need a polynomial inx  because exp_integrate relies on the coefficients for such
        a polynomials.
        We check that shift and mirror yield a result that behaves in a correct way when using exp_int"""
        x = self.x
        poly = PolyExp(sym.Poly([3,1],x),0)
        poly = poly.shift(1)
        diff = testdiff(poly.exp_integrate(1), sym.exp(x)*(3*x-5))
        self.assertEqual(diff,0)
        
        poly = PolyExp(sym.Poly([3,1],x),0)
        poly = poly.mirror(1)
        diff = testdiff(poly.exp_integrate(1), sym.exp(x)*(10-3*x))
        self.assertEqual(diff,0)
    



class TestPiecewisePolyExp(ut.TestCase):
    def setUp(self):
        self.x = x
        self.delta = PiecewisePolyExp(2,arg=self.x)
        self.step = PiecewisePolyExp(0,left_offset=0,right_offset=1,delta_prefac=0,arg=self.x)
        self.example = PiecewisePolyExp(jumpPos=-1,left_offset=1,right_offset=-2,
                                        left_polys=[PolyExp([1,2,3],-4),PolyExp([2,-3,1],0)],
                                        right_polys=[PolyExp([3,1],2)],arg=self.x)
        
    def test_expression_conversion(self):
        x = self.x
        diff = testdiff(self.delta, sym.DiracDelta(x-2))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.step, sym.Heaviside(x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example, sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2))
        self.assertEqual(diff,0)
        
    def test_add_left_poly(self):
        x = self.x
        
        self.delta.add_left_poly(PolyExp([3,2,1],5))
        diff = testdiff(self.delta, sym.DiracDelta(x-2) + sym.Heaviside(-(x-2))*sym.exp(5*x)*(3*x*x +2*x +1))
        self.assertEqual(diff,0)
        
        self.example.add_left_poly(PolyExp([2,0,0],-4))
        diff = testdiff(self.example, sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(3*x*x +2*x+3) + (2*x*x-3*x+1) + 1)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2))
        self.assertEqual(diff,0)
        
    def test_add_right_poly(self):
        x = self.x
        
        self.delta.add_right_poly(PolyExp([3,2,1],5))
        diff = testdiff(self.delta, sym.DiracDelta(x-2) + sym.Heaviside(x-2)*sym.exp(5*x)*(3*x*x +2*x +1))
        self.assertEqual(diff, 0)
        
        self.example.add_right_poly(PolyExp([2,0,0],2))
        diff = testdiff(self.example, sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(2*x*x+3*x+1)-2))
        self.assertEqual(diff,0)
        
    def test_multiplication(self):
        x = self.x
        diff = testdiff(5*self.delta, 5*sym.DiracDelta(x-2))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.delta*5, 5*sym.DiracDelta(x-2))
        self.assertEqual(diff,0)
        
        diff = testdiff(3*self.example, 3*sym.DiracDelta(x+1) + 
                        3*sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)
                       +3*sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2))
        self.assertEqual(diff, 0)
        
    def test_exponential_multiplication(self):
        x = self.x
        diff = testdiff(self.delta.exp_mul(3),sym.exp(6)*sym.DiracDelta(x-2))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.step.exp_mul(-2),sym.exp(-2*x)*sym.Heaviside(x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example.exp_mul(4), sym.DiracDelta(x+1)*sym.exp(-4) 
                        + sym.Heaviside(-(x+1))*(sym.exp(4*x)*(2-3*x+2*x*x) + 3+2*x+x*x)
                       + sym.Heaviside(x+1)*sym.exp(4*x)*(-2+sym.exp(2*x)*(1+3*x)))
        self.assertEqual(diff, 0)
        
    def test_division(self):
        x = self.x
        
        diff = testdiff(self.delta/4, sym.DiracDelta(x-2)/4)
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.example/2, sym.DiracDelta(x+1)/2 + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)/2
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2)/2)
        self.assertEqual(diff,0)
        
    def test_mult_div_with_non_constant(self):
        with self.assertRaises(ValueError):
            self.step*sym.exp(5*self.x)
        with self.assertRaises(ValueError):
            self.example/sym.exp(5*self.x)
            
    def test_addition(self):
        x = self.x
        
        s = self.example + PiecewisePolyExp(-1,left_offset=1,right_offset=2,delta_prefac=3,
                                           left_polys=[PolyExp([2,0],-4)],
                                           right_polys=[PolyExp([3,0,0],1)])
        diff = testdiff(s, 4*sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +4*x+3) + (2*x*x-3*x+1) + 2)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1) + sym.exp(x)*3*x*x))
        self.assertEqual(diff, 0)
        
        self.assertTrue(isinstance(self.step + self.delta + self.example, MultiPiecePolyExp))

    def test_scalar_addition(self):
        x = self.x
        s = self.example + 5
        diff =testdiff(s, sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1 +5)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2 +5) )
        self.assertEqual(diff, 0)
            
            
    def test_subtraction(self):
        x = self.x
        
        d = self.example - PiecewisePolyExp(-1,left_offset=1,right_offset=2,delta_prefac=3,
                                           left_polys=[PolyExp([2,0],-4)],
                                           right_polys=[PolyExp([3,0,0],1)])
        diff = testdiff(d, -2*sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +3) + (2*x*x-3*x+1) )
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1) - sym.exp(x)*3*x*x -4))
        self.assertEqual(diff,0)
        
        self.assertTrue(isinstance(self.step - self.delta - self.example, MultiPiecePolyExp))

    def test_scalar_subtraction(self):
        x = self.x
        s = self.example - 5
        diff =testdiff(s, sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1 -5)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2 -5) )
        self.assertEqual(diff, 0)
            
    def test_exp_integrate(self):
        x = self.x
        
        diff = testdiff(self.delta.exp_integrate(3), sym.exp(6)*sym.Heaviside(x-2) )
        self.assertEqual(diff ,0)
        
        diff = testdiff(self.step.exp_integrate(3), (sym.exp(3*x)-1)*sym.Heaviside(x)/3)
        self.assertEqual(diff, 0)
        diff = testdiff(self.example.exp_integrate(4),sym.Heaviside(x+1)*(5+18*sym.exp(2) -6*sym.exp(6+4*x)
                                                                          +sym.exp(6+6*x)*(1+6*x))/(12*sym.exp(6))
                        + sym.Heaviside(-(x+1))*(7/3 - 9*sym.exp(-4)/4 + x*(9+3*x+x*x)/3 + sym.exp(4*x)*(3-4*x+2*x*x)/4)
                       )
        self.assertEqual(diff,0)
        
    def test_gauss_integrate(self):
        x = self.x
        
        diff = sym.simplify(self.delta.gauss_integrate() -
                           sym.Heaviside(-2+x)/sym.exp(4))
        self.assertEqual(diff, 0)
        
        diff = sym.simplify(self.step.gauss_integrate() -
                           sym.sqrt(sym.pi)*sym.erf(x)*sym.Heaviside(x)/2)
        self.assertEqual(diff, 0)
        
        a = (sym.Heaviside(1+x)/sym.exp(1)
           + sym.Heaviside(1+x)*(3*sym.exp(-3)-3*sym.exp(-(-2+x)*x)
                +sym.sqrt(sym.pi)*(-2*sym.erf(1) +4*sym.exp(1)*(sym.erf(2)-sym.erf(1-x))
                  -2*sym.erf(x)))/2
           - sym.Heaviside(-1-x)*(sym.Rational(5,2)/sym.exp(1) + sym.exp(3)/2 - 3*sym.exp(-x*x)/2
                      + x*sym.exp(-x*x) +sym.exp(-x*(4+x))*x/2 
                        + sym.sqrt(sym.pi)*(-3*sym.erf(1)/2 + 
                             7*sym.exp(4)*sym.erf(1)/4
                            - 3*sym.erf(x)/2
                            -7*sym.exp(4)*sym.erf(2+x)/4)))
        diff = sym.simplify(self.example.gauss_integrate() - a)
        self.assertEqual(diff,0)

    def test_xscale(self):
        x = self.x
        a = sym.Symbol("a")
        
        diff = testdiff(self.delta.xscale(a), sym.DiracDelta(x-2/a)/a)
        self.assertEqual(diff,0)
        
        diff = testdiff(self.step.xscale(-4), sym.Heaviside(-x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example.xscale(a),
                       sym.DiracDelta(x+1/a)/a
                        + sym.Heaviside(x+1/a)*(sym.exp(2*a*x)*(3*a*x+1) -2)
                       + sym.Heaviside(-x-1/a)*(2*a*a*x*x-3*a*x+2+sym.exp(-4*a*x)*(
                       a*a*x*x +2*a*x +3)))
        self.assertEqual(diff, 0)
        
    def test_shift(self):
        x = self.x
        
        diff = testdiff(self.delta.shift(3),sym.DiracDelta(x-5) )
        self.assertEqual(diff,0)
    
        diff = testdiff(self.step.shift(-4), sym.Heaviside(x+4))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example.shift(1), sym.DiracDelta(x) 
                        + sym.Heaviside(-x)*(sym.exp(-4*(-1+x))*(2+x*x)+ 7-7*x+2*x*x)
                        + sym.Heaviside(x)*(-2+sym.exp(2*(x-1))*(3*x-2)))
        self.assertEqual(diff,0)
        
    def test_mirror(self):
        x = self.x
        
        diff = testdiff(self.delta.mirror(2), sym.DiracDelta(x-2))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.step.mirror(-5), sym.Heaviside(-10-x))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.example.mirror(-2), sym.DiracDelta(3+x) 
                        - sym.Heaviside(-3-x)*sym.exp(-2*(4+x))*(11+2*sym.exp(8+2*x) + 3*x)
                       + sym.Heaviside(3+x)*(46+19*x+2*x*x + sym.exp(4*(4+x))*(11+6*x+x**2)))
        self.assertEqual(diff, 0)
        
        
    def test_mirror_shift(self):
        x = self.x
        
        pnew = self.step.shift_mirror(2,1)
        ptest = self.step.shift(-1).mirror(1)
        diff = testdiff(pnew,ptest.as_expr())
        self.assertEqual(diff,0)
        
        xl = sym.floor((random.random()*20-10)*10000)/10000
        xr = sym.floor((random.random()*20-10)*10000)/10000

        pnew = self.example.shift_mirror(xl,xr)
        ptest = self.example.shift(xr-xl).mirror(xr)
        diff = testdiff(pnew, ptest.as_expr())
        self.assertEqual(diff, 0)
        
        
class TestMultiPiecePolyExp(ut.TestCase):
    def setUp(self):
        self.x = x
        self.delta = MultiPiecePolyExp([PiecewisePolyExp(2,arg=self.x)])
        self.step = MultiPiecePolyExp([PiecewisePolyExp(0,left_offset=0,right_offset=1,delta_prefac=0,arg=self.x)])
        self.example = MultiPiecePolyExp([PiecewisePolyExp(jumpPos=-1,left_offset=1,right_offset=-2,
                                        left_polys=[PolyExp([1,2,3],-4),PolyExp([2,-3,1],0)],
                                        right_polys=[PolyExp([3,1],2)],arg=self.x)])
        

    ## Copies of tests for PiecewisePolyExp
    def test_expression_conversion(self):
        x = self.x
        diff = testdiff(self.delta, sym.DiracDelta(x-2))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.step, sym.Heaviside(x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example, sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2))
        self.assertEqual(diff,0)
        
        
        
    def test_multiplication(self):
        x = self.x
        diff = testdiff(5*self.delta, 5*sym.DiracDelta(x-2))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.delta*5, 5*sym.DiracDelta(x-2))
        self.assertEqual(diff,0)
        
        diff = testdiff(3*self.example, 3*sym.DiracDelta(x+1) + 
                        3*sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)
                       +3*sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2))
        self.assertEqual(diff, 0)
        
    def test_exponential_multiplication(self):
        x = self.x
        diff = testdiff(self.delta.exp_mul(3),sym.exp(6)*sym.DiracDelta(x-2))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.step.exp_mul(-2),sym.exp(-2*x)*sym.Heaviside(x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example.exp_mul(4), sym.DiracDelta(x+1)*sym.exp(-4) 
                        + sym.Heaviside(-(x+1))*(sym.exp(4*x)*(2-3*x+2*x*x) + 3+2*x+x*x)
                       + sym.Heaviside(x+1)*sym.exp(4*x)*(-2+sym.exp(2*x)*(1+3*x)))
        self.assertEqual(diff, 0)
        
    def test_division(self):
        x = self.x
        
        diff = testdiff(self.delta/4, sym.DiracDelta(x-2)/4)
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.example/2, sym.DiracDelta(x+1)/2 + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +2*x+3) + (2*x*x-3*x+1) + 1)/2
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1)-2)/2)
        self.assertEqual(diff,0)
        
    def test_mult_div_with_non_constant(self):
        with self.assertRaises(ValueError):
            self.step*sym.exp(5*self.x)
        with self.assertRaises(ValueError):
            self.example/sym.exp(5*self.x)
            
    def test_addition(self):
        x = self.x
        
        s = self.example + PiecewisePolyExp(-1,left_offset=1,right_offset=2,delta_prefac=3,
                                           left_polys=[PolyExp([2,0],-4)],
                                           right_polys=[PolyExp([3,0,0],1)])
        diff = testdiff(s, 4*sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +4*x+3) + (2*x*x-3*x+1) + 2)
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1) + sym.exp(x)*3*x*x))
        self.assertEqual(diff,0)

    def test_add_constant(self):
        x = self.x

        p = MultiPiecePolyExp([PiecewisePolyExp(2,arg=self.x),PiecewisePolyExp(0,left_offset=0,right_offset=1,delta_prefac=0,arg=self.x)])
        s = p + 5

        diff = testdiff(s, sym.DiracDelta(x-2) + sym.Heaviside(x)
                        + 5*(sym.Heaviside(x-2)+sym.Heaviside(-x+2)))
        self.assertEqual(diff,0)

            
    def test_subtraction(self):
        x = self.x
        
        d = self.example - PiecewisePolyExp(-1,left_offset=1,right_offset=2,delta_prefac=3,
                                           left_polys=[PolyExp([2,0],-4)],
                                           right_polys=[PolyExp([3,0,0],1)])
        diff = testdiff(d, -2*sym.DiracDelta(x+1) + 
                        sym.Heaviside(-(x+1))*(sym.exp(-4*x)*(x*x +3) + (2*x*x-3*x+1) )
                       +sym.Heaviside(x+1)*(sym.exp(2*x)*(3*x+1) - sym.exp(x)*3*x*x -4))
        self.assertEqual(diff,0)

    def test_subtract_constant(self):
        x = self.x

        p = MultiPiecePolyExp([PiecewisePolyExp(2,arg=self.x),PiecewisePolyExp(0,left_offset=0,right_offset=1,delta_prefac=0,arg=self.x)])
        s = p - 5

        diff = testdiff(s, sym.DiracDelta(x-2) + sym.Heaviside(x)  
                        - 5*(sym.Heaviside(x-2)+sym.Heaviside(-x+2)))
        self.assertEqual(diff,0)
        
    def test_exp_integrate(self):
        x = self.x
        
        diff = testdiff(self.delta.exp_integrate(3), sym.exp(6)*sym.Heaviside(x-2) )
        self.assertEqual(diff ,0)
        
        diff = testdiff(self.step.exp_integrate(3), (sym.exp(3*x)-1)*sym.Heaviside(x)/3)
        self.assertEqual(diff, 0)
        diff = testdiff(self.example.exp_integrate(4),sym.Heaviside(x+1)*(5+18*sym.exp(2) -6*sym.exp(6+4*x)
                                                                          +sym.exp(6+6*x)*(1+6*x))/(12*sym.exp(6))
                        + sym.Heaviside(-(x+1))*(7/3 - 9*sym.exp(-4)/4 + x*(9+3*x+x*x)/3 + sym.exp(4*x)*(3-4*x+2*x*x)/4)
                       )
        self.assertEqual(diff,0)
        
    def test_gauss_integrate(self):
        x = self.x
        
        diff = sym.simplify(self.delta.gauss_integrate() -
                           sym.Heaviside(-2+x)/sym.exp(4))
        self.assertEqual(diff, 0)
        
        diff = sym.simplify(self.step.gauss_integrate() -
                           sym.sqrt(sym.pi)*sym.erf(x)*sym.Heaviside(x)/2)
        self.assertEqual(diff, 0)
        
        a = (sym.Heaviside(1+x)/sym.exp(1)
           + sym.Heaviside(1+x)*(3*sym.exp(-3)-3*sym.exp(-(-2+x)*x)
                +sym.sqrt(sym.pi)*(-2*sym.erf(1) +4*sym.exp(1)*(sym.erf(2)-sym.erf(1-x))
                  -2*sym.erf(x)))/2
           - sym.Heaviside(-1-x)*(sym.Rational(5,2)/sym.exp(1) + sym.exp(3)/2 - 3*sym.exp(-x*x)/2
                      + x*sym.exp(-x*x) +sym.exp(-x*(4+x))*x/2 
                        + sym.sqrt(sym.pi)*(-3*sym.erf(1)/2 + 
                             7*sym.exp(4)*sym.erf(1)/4
                            - 3*sym.erf(x)/2
                            -7*sym.exp(4)*sym.erf(2+x)/4)))
        diff = sym.simplify(self.example.gauss_integrate() - a)
        self.assertEqual(diff,0)

    def test_xscale(self):
        x = self.x
        a = sym.Symbol("a")
        
        diff = testdiff(self.delta.xscale(a), sym.DiracDelta(x-2/a)/a)
        self.assertEqual(diff,0)
        
        diff = testdiff(self.step.xscale(-4), sym.Heaviside(-x))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example.xscale(a),
                       sym.DiracDelta(x+1/a)/a
                        + sym.Heaviside(x+1/a)*(sym.exp(2*a*x)*(3*a*x+1) -2)
                       + sym.Heaviside(-x-1/a)*(2*a*a*x*x-3*a*x+2+sym.exp(-4*a*x)*(
                       a*a*x*x +2*a*x +3)))
        self.assertEqual(diff, 0)
        
    def test_shift(self):
        x = self.x
        
        diff = testdiff(self.delta.shift(3),sym.DiracDelta(x-5) )
        self.assertEqual(diff,0)
    
        diff = testdiff(self.step.shift(-4), sym.Heaviside(x+4))
        self.assertEqual(diff,0)
        
        diff = testdiff(self.example.shift(1), sym.DiracDelta(x) 
                        + sym.Heaviside(-x)*(sym.exp(-4*(-1+x))*(2+x*x)+ 7-7*x+2*x*x)
                        + sym.Heaviside(x)*(-2+sym.exp(2*(x-1))*(3*x-2)))
        self.assertEqual(diff,0)
        
    def test_mirror(self):
        x = self.x
        
        diff = testdiff(self.delta.mirror(2), sym.DiracDelta(x-2))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.step.mirror(-5), sym.Heaviside(-10-x))
        self.assertEqual(diff, 0)
        
        diff = testdiff(self.example.mirror(-2), sym.DiracDelta(3+x) 
                        - sym.Heaviside(-3-x)*sym.exp(-2*(4+x))*(11+2*sym.exp(8+2*x) + 3*x)
                       + sym.Heaviside(3+x)*(46+19*x+2*x*x + sym.exp(4*(4+x))*(11+6*x+x**2)))
        self.assertEqual(diff, 0)
        
        
    def test_mirror_shift(self):
        x = self.x
        
        pnew = self.step.shift_mirror(2,1)
        ptest = self.step.shift(-1).mirror(1)
        diff = testdiff(pnew,ptest.as_expr())
        self.assertEqual(diff,0)
        
        xl = sym.floor((random.random()*20-10)*10000)/10000
        xr = sym.floor((random.random()*20-10)*10000)/10000

        pnew = self.example.shift_mirror(xl,xr)
        ptest = self.example.shift(xr-xl).mirror(xr)
        diff = testdiff(pnew, ptest.as_expr())
        self.assertEqual(diff, 0)

    ## Tests for unique functionality of MultiPiecePolyExp





class TestGaussIntegrate(ut.TestCase):
    def test_odd_gauss_int_helper(self):
        x = sym.abc.x
        zero_Poly = sym.Poly([],x)
        self.assertEqual(odd_gauss_int_helper(zero_Poly),0)
        
        const_poly = sym.Poly([2],x)
        self.assertEqual(odd_gauss_int_helper(const_poly),-1)
        
        long_poly = sym.Poly([sym.Rational(1,2),-1,2,sym.Rational(5,6),-1,sym.Rational(3,5),
                              sym.Rational(2,3), -sym.Rational(5,3), sym.Rational(10,3),
                              sym.Rational(5,3)],x)
        self.assertEqual(odd_gauss_int_helper(long_poly),sym.Poly([-15,-105,-900,
                                                                   -6325,-37920,-189618,
                                                                   -758492,-2275426,-4550952,
                                                                   -4551002],x)/60)
    
    def test_odd_gauss_int(self):
        x = sym.abc.x
        zero_poly = sym.Poly([],x)
        self.assertEqual(odd_gauss_int(zero_poly),0)
        
        const_poly = sym.Poly([-3],x)
        self.assertEqual(odd_gauss_int(const_poly), sym.Rational(3,2)*sym.exp(-x**2))
        
        long_poly = sym.Poly([8,-1,sym.Rational(3,8),-sym.Rational(2,9)],x)
        diff = sym.simplify(odd_gauss_int(long_poly) + sym.Poly([576,1656,3339,3323],x*x)*sym.exp(-x*x)/144)
        self.assertEqual(diff,0)
    
    def test_even_gauss_int_helper(self):
        x = sym.abc.x
        zero_poly = sym.Poly([],x)
        self.assertRaises(ValueError)
        
        const_poly = sym.Poly([4],x)
        self.assertEqual(even_gauss_int_helper(const_poly), (-2*x,1))
        
        long_poly = sym.Poly([sym.Rational(1,5),sym.Rational(3,7),
                                        sym.Rational(1,3),-sym.Rational(1,7)],x)
    
        res_poly, erf_fac = even_gauss_int_helper(long_poly)
        diff = sym.simplify(res_poly.as_expr() - x*sym.Poly([-sym.Rational(1,10),-sym.Rational(79,140),
                                      -sym.Rational(265,168),-sym.Rational(257,112)],x*x).as_expr())
        self.assertEqual(diff,0)
        self.assertEqual(erf_fac,sym.Rational(257,224))
        
    def test_even_gauss_int(self):
        x = sym.abc.x
        zero_poly = sym.Poly([],x)
        self.assertEqual(even_gauss_int(zero_poly),0)
        const_poly = sym.Poly([5],x)
        self.assertEqual(even_gauss_int(const_poly),5*sym.sqrt(sym.pi)*sym.erf(x)/2)
        
        long_poly = sym.Poly([2,-sym.Rational(6,7),-sym.Rational(1,3),-sym.Rational(7,10)], x)
        diff = sym.simplify(even_gauss_int(long_poly) + x*(247+174*x*x+84*x**4)*sym.exp(-x*x)/84 
                            - 941*sym.sqrt(sym.pi)*sym.erf(x)/840)
        self.assertEqual(diff,0)

class TestTimeEvolution(ut.TestCase):
    def test_higher_order_poly(self):
        test_poly = PiecewisePolyExp(jumpPos=0,right_polys=[PolyExp([3,2,1],3,arg=x)])
        time_evolution(test_poly,lower=-5,upper=10)

    @ut.skip
    def test_higher_order_poly_infinty(self):
        inf_test_step = PiecewisePolyExp(jumpPos=0,delta_prefac=0,right_polys=[PolyExp([1,0],1)])
        te_inf = time_evolution(inf_test_step,lower=-sym.oo, upper=sym.oo)
        self.assertNotEqual(te_inf,sym.nan)
        print(te_inf)

    def test_empty_MultiPiece(self):
        p = MultiPiecePolyExp()
        te = time_evolution(p)
        self.assertEqual(te, sym.sympify(0))


class TestMirrorOperators(ut.TestCase):
    def setUp(self):
        x0 = x0n
        self.fl = PiecewisePolyExp(x0,delta_prefac=sym.exp(-f0*x0/2))
        self.fr = PiecewisePolyExp(-x0,delta_prefac=0)

    def test_MirrorRight_singleStep(self):
        fl2 = mirrorRight(self.fl, self.fr, w=w,f0=f0,xlp=0,xr=0,x0=0,fl_start=0)
        fl2_res = '(f*exp(f*x_0*tanh(w/2)/2)*tanh(w/2)**2 + f*exp(f*x_0*tanh(w/2)/2)*tanh(w/2))*exp(-f*x_0/2)*exp(f*x*tanh(w/2)/2)*Heaviside(x + x_0)/2 + exp(-f*x_0/2)*tanh(w/2)*DiracDelta(x + x_0)'
        diff = sym.simplify(fl2.as_expr() - sym.sympify(fl2_res).subs([("x_0",x0n), ("f",f0),
                                                                       ("w",w), ("x",x)]))
        self.assertEqual(diff, 0)

    def test_MirrorLeft_singleStep(self):
        fr2 = mirrorLeft(self.fl, self.fr, w=w,f0=f0,xlp=0,xr=0,x0=0, fr_start=0)
        fr2_res = '(1 - tanh(w/2))*exp(-f*x_0/2)*DiracDelta(x - x_0) - (f*exp(-f*x_0/2)*exp(f*x_0*tanh(w/2)/2)*tanh(w/2)**2/2 - f*exp(-f*x_0/2)*exp(f*x_0*tanh(w/2)/2)*tanh(w/2)/2)*exp(-f*x*tanh(w/2)/2)*Heaviside(-x + x_0)' 
        diff = sym.simplify(fr2.as_expr() - sym.sympify(fr2_res).subs([("x_0",x0n), ("f",f0),
                                                                       ("w",w), ("x",x)]))
        self.assertEqual(diff, 0)

    def test_Mirror_functions_consistency(self):
        fl2_regular = mirrorRight(self.fl, self.fr, w=w,f0=f0,xlp=0,xr=0,x0=0, fl_start=0)
        fl2_symmetry = mirrorLeft(self.fr.mirror(0), self.fl.mirror(0),
                                  w=-w,f0=-f0,xlp=0,xr=0,x0=0,fr_start=0).mirror(0)
        diff = sym.simplify(str(fl2_regular - fl2_symmetry))
        self.assertEqual(diff, 0)

        fr2_regular = mirrorLeft(self.fl, self.fr, w=w,f0=f0,xlp=0,xr=0,x0=0, fr_start=0)
        fr2_symmetry = mirrorRight(self.fr.mirror(0), self.fl.mirror(0),
                                  w=-w,f0=-f0,xlp=0,xr=0,x0=0, fl_start=0).mirror(0)
        diff = sym.simplify(str(fr2_regular - fr2_symmetry))
        self.assertEqual(diff, 0)

class TestNumerics(ut.TestCase):
    def test_numerical_time_evolution(self):
        test_poly = PiecewisePolyExp(jumpPos=0,right_polys=[PolyExp([3,2,1],3,arg=x)])
        te = time_evolution(test_poly,lower=-5,upper=10)
        num_te = gen_numerical_time_evolution(te, 2.)
        num_te(2,1)



if __name__ == "__main__":
    ut.main()
