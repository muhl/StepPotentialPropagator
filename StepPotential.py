#!/usr/bin/env python
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


import sympy as sym
from sympy.core.numbers import NegativeInfinity
as_poly = sym.Poly.as_poly
import sympy.abc
from sympy import poly
import copy
import random
import mpmath as mpm
import scipy as sp
import scipy.sparse
import scipy.integrate
import numpy as np
from collections import deque

# Symbol definitions

f0, w = sym.symbols("f w", finite=True, real=True)
x0n = sym.Symbol("x_0", finite=True, real=True, negative=True)
x0p = sym.Symbol("x_0", finite=True, real=True, positive=True)
x = sym.Symbol("x", finite=True, real=True)
xp = sym.Symbol("x'", finite=True, real=True)
t = sym.Symbol("t", finite=True, real=True, positive=True)

class PolyExp:
    """representation of terms of the form :math:`f(x) \equiv C \cdot p(x)exp(a \cdot x)`, where p(x) is some arbitrary
    polynomial with known coefficients and a is some real constant. The class
    implements basic algebraic operations that are needed to perform integrals
    that are needed for the mirroring technique described in the paper. 
    Supported operations are:

    - addition/subtraction of multiple PolyExp object with the same
      exponential slope
    - multiplication/division with constant factors

    Args:
        poly (sympy.polys.polytools.poly): Polynomial. Can be either:

            - a list of prefactors sorted from highest to lowest order and zeroes
              indicating missing terms, or
            - a sympy.Poly object that depends on the same argument as is given
              as ``arg`` parameter

        exp_slope (float): parameter :math:`a` in the exponential function
        prefac (float): constant prefactor :math:`C`.
        arg (sympy.core.symbol.Symbol): argument of the function. Defaults to :math:`x` that
            is defined as finite and real (see constants defined by this module)
    """
    def __init__(self, poly, exp_slope,prefac=1,arg=x):
        if arg in sym.sympify(prefac).free_symbols:
            raise ValueError("prefactor must be constant")
        if isinstance(poly,sym.Poly):
            self.poly = poly*as_poly(prefac,poly.gens)
        elif isinstance(poly, list):
            self.poly = sym.Poly.from_list(poly,arg)*as_poly(prefac,arg)
        else:
            self.poly = sym.Poly(prefac*poly,arg)
        self.exp_slope = exp_slope
        self.x = arg
        assert isinstance(self.poly,sym.Poly)
        
    def as_expr(self):
        """converts PolyExp to a standard sympy expression"""
        return sym.exp(self.exp_slope*self.x)*self.poly.as_expr()
    def __repr__(self):
        return self.as_expr().__repr__()
    
    def _repr_latex_(self):
        return self.as_expr()._repr_latex_()
    
    def __mul__(self,other):
        """multiplication routine
        PolyExp objects can be multiplied by any sympy expression that does not
        contain the argument of the PolyExp itself. The multiplying factor will
        be applied to the prefactor and a new PolyExp is returned"""
        if self.x in sym.sympify(other).free_symbols:
            raise ValueError("can only multiply with constant factors")
        res = PolyExp(self.poly,self.exp_slope,other,self.x)
        assert isinstance(res.poly,sym.Poly)
        return res
    
    def __rmul__(self,other):
        """right handed multiplication is defined such that multiplication is 
        commutative"""
        return self.__mul__(other)
    
    def __truediv__(self,other):
        if self.x in sym.sympify(other).free_symbols:
            raise ValueError("can only divide by constant factors")
        res =  PolyExp(self.poly,self.exp_slope,1/other,self.x)
        assert isinstance(res.poly, sym.Poly)
        return res
        
    def __add__(self,other):
        if not isinstance(other,PolyExp):
            raise TypeError()
        if other.exp_slope != self.exp_slope or other.x != self.x:
            raise ValueError("can only add PolyExp with the same decay rate and argument")
        return PolyExp(self.poly+other.poly,self.exp_slope,prefac=1, arg=self.x)
    
    def __sub__(self,other):
        if not isinstance(other,PolyExp):
            raise TypeError()
        if other.exp_slope != self.exp_slope or other.x != self.x:
            raise ValueError("can only subtract PolyExp with the same decay rate and argument")
        return PolyExp(self.poly-other.poly,self.exp_slope,prefac=1, arg=self.x)
    
    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return self
    
    def exp_mul(self, exp_slope):
        """multiply PolyExp with :math:`\exp( a \cdot x)` thus adding :math:`a`
        to the exponential slope of the object.

        Args:
            exp_slope(float): 

        Returns:
            PolyExp: copy of the object with changed exponential slope
        """
        return PolyExp(sym.Poly.from_poly(self.poly),self.exp_slope+exp_slope,1,self.x)
    
    def exp_integrate(self,b):
        r"""calculate the exponential integral :math:`\int f(x) \exp(b \cdot x)`

        Args:
            b(float): parameter of the exponential integration kernel

        Returns:
            PolyExp: result.

        The result of the integral is of the form :math:`\tilde{p}(x) \exp((a+b) \cdot x)`
        with a new polynomial :math:`\tilde{p}(x)`
        """
        a = self.exp_slope + b
        if a != 0:
            coeffs = self.poly.all_coeffs()[::-1]
            n=self.poly.degree()
            if isinstance(n, NegativeInfinity):
                return PolyExp(sym.Poly(0,self.x),0)
            coeffs_new = []
            for i in range(n+1):
                ci = 0
                for j in range(i+1):
                    fac = 1
                    for k in range(n-i+1,n-i+j+1):
                        fac *= k
                    ci += coeffs[n-i+j]*fac*(-1)**j/a**(j+1)
                coeffs_new.append(ci)
            return PolyExp(sym.Poly.from_list(coeffs_new,self.x),a,prefac=1)
        else:
            return PolyExp(sym.Poly(self.poly.integrate(), self.poly.args[1]),0,prefac=1)
        
    def gauss_integrate(self):
        r"""calculate the gaussian integral :math:`\int f(x) \exp(-x^2)`

        Returns:
            sympy.core.expr.Expr: result
        """
        x = self.x
        shifted_poly = sym.Poly(self.poly.subs(x,x+self.exp_slope*sym.Rational(1,2)),self.poly.args[1])
        "workaround for linear polynomials"
        shifted_poly = sym.Poly.from_poly(shifted_poly,x)
        new_prefac = sym.exp(self.exp_slope**2*sym.Rational(1,4))
        coeffs = shifted_poly.all_coeffs()

        even_coeffs = coeffs[-1::-2][::-1]
        odd_coeffs = coeffs[-2::-2][::-1]
        
        
        res = new_prefac*sym.collect(even_gauss_int(sym.Poly(even_coeffs,self.x)) 
                              +odd_gauss_int(sym.Poly(odd_coeffs,self.x)),
                            sym.exp(-self.x**2)
                        ).subs(x,x-self.exp_slope*sym.Rational(1,2))
        return res.expand().collect(sym.exp(self.exp_slope*x)*sym.exp(-x*x)).collect(
            sym.erf(x-self.exp_slope*sym.Rational(1,2)))
        
        
    def evaluate(self,x0):
        r"""evaluate the function a given point

        Args:
            x0(float): :math:`x`-value

        Returns:
            sympy.core.expr.Expr: :math:`f(x_0)`
        """
        return self.as_expr().subs(self.x,x0)
    
    def xscale(self, fac):
        r"""scale argument of the function by factor

        Args:
            fac(float): scaling factor

        Returns:
            PolyExp: copy of the original function with different 
            scaling :math:`f(\text{fac} \cdot x)`
        """
        pnew = self.copy()
        pnew.poly = sym.Poly.from_poly(pnew.poly.subs(pnew.x,fac*pnew.x),pnew.x)
        pnew.exp_slope *= fac
        return pnew
        
            
    def shift(self,shift):
        r"""shifts function by given amount

        Args:
            shift(float): amount of x-shift

        Returns:
            PolyExp: copy of function with shifted argument :math:`f(x-\text{shift})`
        """
        pnew = self.copy()
        x = pnew.x
        pnew.poly = sym.Poly(pnew.poly.subs(pnew.x,pnew.x-shift),pnew.poly.args[1])
        "workaround for linear polynomials"
        if pnew.poly.degree()<=1:
            pnew.poly = sym.Poly.from_poly(pnew.poly,x)
        pnew *= sym.exp(-pnew.exp_slope*shift)
        assert isinstance(pnew.poly, sym.Poly)
        return pnew
    
    def mirror(self,pos):
        r"""mirrors function around given point

        Args:
            pos(float): mirror position

        Returns:
            PolyExp: mirrored function :math:`f(2 \cdot \text{pos} - x)`
        """

        pnew = self.copy()
        x = pnew.x
        pnew.poly = sym.Poly(pnew.poly.subs(pnew.x,2*pos-pnew.x),pnew.x)
        if pnew.poly.degree() <= 1:
            pnew.poly = sym.Poly.from_poly(pnew.poly,x)
        pnew *= sym.exp(2*pos*pnew.exp_slope)
        pnew.exp_slope *= -1
        assert isinstance(pnew.poly, sym.Poly)
        return pnew
    
    def shift_mirror(self,xl,xr):
        r"""combination of shift and mirror operation for use at periodic
        boundaries

        Args:
            xl(float): postion of mirror for the left interval
            xp(float): position of mirror for the right interval

        Returns:
            PolyExp: mirrored and shifted function
            :math:`f(x_\text{l} + x_\text{r} - x)`
        """
        pnew = self.copy()
        x = pnew.x
        pnew.poly = sym.Poly(pnew.poly.subs(pnew.x,xr+xl - pnew.x),pnew.poly.args[1])
        if pnew.poly.degree() <= 1:
            pnew.poly = sym.Poly.from_poly(pnew.poly, x)
        pnew *= sym.exp((xl+xr)*pnew.exp_slope)
        pnew.exp_slope *= -1
        assert isinstance(pnew.poly, sym.Poly)
        return pnew
    
    def copy(self):
        """returns a independent deepcopy of the PolyExp object"""
        return PolyExp(sym.Poly.from_poly(self.poly),self.exp_slope,1,self.x)


class PiecewisePolyExp:
    r"""
    representation of terms of the form
    :math:`\left(\sum_i p_{\text{l},i}(x) \exp(a_{\text{l},i} x) c_\text{l} \right) \theta(x-x_0) + \left(\sum_i p_{\text{r},i}(x) \exp(a_{\text{r},i} x) + c_\text{r}\right) \theta(-x+x_0) + b \delta(x-x_0)`

    Supported algebraic operations are:

    - multiplication/division with constant factor
    - addition/subtraction of two PiecewisePolyExp with the same jump position
      will yield a PiecewisePolyExp
    - addition/subtraction of two PiecewisePolyExp with different jump
      positions will yield a MultiPiecePolyExp
    - addition/subtraction of constant terms
    - addition/subtraction of PolyExp objects

    Args:
        jumpPos(float): position of the Dirac-Delta function and flanks of step
            functions :math:`x_0`
        left_offset(float): offset of the left part :math:`c_\text{l}`
        right_offset(float): offset of the left part :math:`c_\text{r}`
        left_polys(list(PolyExp)): list of PolyExp of the left part
        right_polys(list(PolyExp)): list of PolyExp of the right part
        delta_prefac(float): prefactor of the Dirac Delta function :math:`b`
        arg(sympy.core.symbol.Symbol): argument of the function. Make sure that it is the
            same as for all PolyExp 
    """
    def __init__(self, jumpPos, left_offset = 0, right_offset = 0,left_polys=None,right_polys=None ,delta_prefac = 1,arg=x):
        self.jumpPos = jumpPos
        self.left_polys = {}
        self.right_polys = {}
        if left_polys is not None:
            for poly in left_polys:
                self.add_left_poly(poly)
                
        if right_polys is not None:
            for poly in right_polys:
                self.add_right_poly(poly)
            
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.delta_prefac = delta_prefac
        self.x=arg
        
        
    @staticmethod
    def zero(jumpPos, arg=x):
        r"""
        construct an empty PiecewisePolyExp with given jump position

        Args:
            jumpPos(float): position of the Dirac Delta Function and step functions
            arg(sym.Symbol): argument of the function

        Returns
            PiecewisePolyExp
        """
        return PiecewisePolyExp(jumpPos,delta_prefac=0,arg=arg)
        
    def as_expr(self):
        r"""representation of the piecewise function as a regular sympy expression

        Returns:
            sympy.core.expr.Expr: :math:`f(x)`
        """
        res = self.delta_prefac*sym.DiracDelta(self.x-self.jumpPos)
        res += (sum(poly.as_expr() for poly in self.get_left_polys()) + self.left_offset)*sym.Heaviside(-(self.x-self.jumpPos))
        res += (sum(poly.as_expr() for poly in self.get_right_polys()) + self.right_offset)*sym.Heaviside((self.x-self.jumpPos))
        return res
    
    def evaluate(self, x0):
        """evaluate PiecewisePolyExp at given position

        Args:
            x0(float): x-value

        Returns:
            sympy.core.expr.Expr: :math:`f(x_0)`
        """
        return self.as_expr().subs(self.x,x0)
    
    def __call__(self, x0):
        return self.evaluate(x0)

    def add_left_poly(self,poly):
        """add a PolyExp term to the left interval

        Args:
            poly(PolyExp): PolyExp to add
        """
        if poly.exp_slope in self.left_polys:
            self.left_polys[poly.exp_slope] += poly
        else:
            self.left_polys[poly.exp_slope] = poly
    
    def add_right_poly(self,poly):
        """add a PolyExp term to the right interval

        Args:
            poly(PolyExp): PolyExp to add
        """
        if poly.exp_slope in self.right_polys:
            self.right_polys[poly.exp_slope] += poly
        else:
            self.right_polys[poly.exp_slope] = poly
    
    def get_left_polys(self):
        """return a list of the PolyExp object for the left interval"""
        return list(self.left_polys.values())
    
    def get_right_polys(self):
        """return a list of the PolyExp object for the right interval"""
        return list(self.right_polys.values())
        
    def __repr__(self):
        return self.as_expr().__repr__()
    
    def _repr_latex_(self):
        return self.as_expr()._repr_latex_()
    
    def __mul__(self, other):
        if self.x in sym.sympify(other).free_symbols:
            raise ValueError("can only multiply with constant factors")
        return PiecewisePolyExp(self.jumpPos, 
                               delta_prefac=self.delta_prefac*other,
                              left_offset=self.left_offset*other,
                              right_offset=self.right_offset*other,
                              left_polys=[poly*other for poly in self.get_left_polys()],
                              right_polys=[poly*other for poly in self.get_right_polys()]
                               )
    
    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return self
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __truediv__(self,other):
        if self.x in sym.sympify(other).free_symbols:
            raise ValueError("can only divide by constant factors")
        return PiecewisePolyExp(self.jumpPos, 
                               delta_prefac=self.delta_prefac/other,
                              left_offset=self.left_offset/other,
                              right_offset=self.right_offset/other,
                              left_polys=[poly/other for poly in self.get_left_polys()],
                              right_polys=[poly/other for poly in self.get_right_polys()]
                               )
    
    def __add__(self, other):
        if isinstance(other, PiecewisePolyExp):
            assert self.x == other.x
            if self.jumpPos == other.jumpPos:
                return PiecewisePolyExp(jumpPos=self.jumpPos,
                               delta_prefac=self.delta_prefac+other.delta_prefac,
                               left_offset=self.left_offset+other.left_offset,
                               right_offset=self.right_offset+other.right_offset,
                               left_polys=self.get_left_polys()+other.get_left_polys(),
                               right_polys=self.get_right_polys()+other.get_right_polys(),
                               arg=self.x)
            else:
                return MultiPiecePolyExp([self, other],arg=self.x)
        elif isinstance(other, MultiPiecePolyExp):
            return other + self
        elif isinstance(other, (int, float, sym.Expr)):
            if isinstance(other, sym.Expr) and self.x in other.free_symbols:
                raise ValueError("can only add constant terms")
            res = self.copy()
            s_other = sym.sympify(other)
            res.left_offset += s_other
            res.right_offset += s_other
            return res
        elif isinstance(other, PolyExp):
            assert self.x == other.x
            res = self.copy()
            res.add_left_poly(other)
            res.add_right_poly(other)
            return res
        else:
            raise TypeError()

    def __radd__(self, other):
        return self + other
            
    
    def __sub__(self,other):
        if isinstance(other, PiecewisePolyExp):
            assert self.x == other.x
            if self.jumpPos == other.jumpPos:
                return PiecewisePolyExp(jumpPos=self.jumpPos,
                           delta_prefac=self.delta_prefac-other.delta_prefac,
                           left_offset=self.left_offset-other.left_offset,
                           right_offset=self.right_offset-other.right_offset,
                           left_polys=self.get_left_polys()+[-poly for poly in other.get_left_polys()],
                           right_polys=self.get_right_polys()+[-poly for poly in other.get_right_polys()])
            else:
                return MultiPiecePolyExp([self,-other],arg=self.x)
        elif isinstance(other, MultiPiecePolyExp):
            return -(other - self)
        elif isinstance(other, (int, float, sym.Expr)):
            if isinstance(other, sym.Expr) and self.x in other.free_symbols:
                raise ValueError("can only add constant terms")
            res = self.copy()
            s_other = sym.sympify(other)
            res.left_offset -= s_other
            res.right_offset -= s_other
            return res
        elif isinstance(other, PolyExp):
            assert self.x == other.x
            res = self.copy()
            res.add_left_poly(-other)
            res.add_right_poly(-other)
            return res
        else:
            raise TypeError()

    def __rsub__(self, other):
        return -(self - other)
    
    def exp_mul(self, exp_slope):
        left_polys = [poly.exp_mul(exp_slope) for poly in self.get_left_polys()]
        if not self.left_offset == 0:
            left_polys.append(PolyExp([self.left_offset],exp_slope,1,self.x))
        right_polys = [poly.exp_mul(exp_slope) for poly in self.get_right_polys()]
        if not self.right_offset == 0:
            right_polys.append(PolyExp([self.right_offset], exp_slope, 1, self.x))
        return PiecewisePolyExp(jumpPos=self.jumpPos, delta_prefac=self.delta_prefac*sym.exp(exp_slope*self.jumpPos),
                              left_polys=left_polys, right_polys=right_polys,arg=self.x)
    
    def exp_integrate(self, b):
        r"""calculate the exponential integral :math:`\int f(x) \exp(b \cdot x)`

        Args:
            b(float): parameter of the exponential integration kernel

        Returns:
            PiecewisePolyExp: result.

        The result of the integral is of the same form as the initial function,
        only the polynomials and exponential prefactors are different.
        """
        res = PiecewisePolyExp(jumpPos=self.jumpPos,delta_prefac=0,
                               right_offset=self.delta_prefac*sym.exp(self.jumpPos*b),arg=self.x)
        left_antider = [poly.exp_integrate(b) for poly in self.get_left_polys()]
        left_antider.append(PolyExp(sym.Poly(self.left_offset,self.x),0).exp_integrate(b))
        right_antider = [poly.exp_integrate(b) for poly in self.get_right_polys()]
        right_antider.append(PolyExp(sym.Poly(self.right_offset,self.x),0).exp_integrate(b))
        
        for poly in left_antider:
            res.add_left_poly(poly)
            
        for poly in right_antider:
            res.add_right_poly(poly)
        
        res.left_offset -= sum(poly.evaluate(self.jumpPos) for poly in left_antider)
        res.right_offset -= sum(poly.evaluate(self.jumpPos) for poly in right_antider)
        
        return res
    
    def gauss_integrate(self):
        r"""calculate the gaussian integral :math:`\int f(x) \exp(-x^2)`

        Returns:
            sympy.core.expr.Expr: result
        """
        x = self.x
        res = self.delta_prefac*sym.exp(-self.jumpPos**2)*sym.Heaviside(x-self.jumpPos)
        left_antider = PolyExp([self.left_offset],exp_slope=0,prefac=1,arg=x).gauss_integrate()
        for poly in self.get_left_polys():
            left_antider += poly.gauss_integrate()
        right_antider = PolyExp([self.right_offset],exp_slope=0,prefac=1,arg=x).gauss_integrate()
        for poly in self.get_right_polys():
            right_antider += poly.gauss_integrate()
            
        res += sym.Heaviside(x-self.jumpPos)*(right_antider-right_antider.subs(x,self.jumpPos))
        res += sym.Heaviside(self.jumpPos-x)*(left_antider-left_antider.subs(x,self.jumpPos))
            
        return res
    
    def xscale(self, fac):
        r"""scale argument of the function by factor

        Args:
            fac(float): scaling factor

        Returns:
            PiecewisePolyExp: copy of the original function with different 
            scaling :math:`f(\text{fac} \cdot x)`
        """
        fac = sym.sympify(fac)
        left_polys = [poly.xscale(fac) for poly in self.get_left_polys()]
        right_polys = [poly.xscale(fac) for poly in self.get_right_polys()]
        
        # check if fac is a symbolic expression, if so we assume it is positive
        if len(fac.free_symbols) != 0 or fac>0:         
            return PiecewisePolyExp(jumpPos=self.jumpPos/fac,
                                   left_offset=copy.deepcopy(self.left_offset),
                                   right_offset=copy.deepcopy(self.right_offset),
                                   left_polys=left_polys,
                                   right_polys=right_polys,
                                   delta_prefac=copy.deepcopy(self.delta_prefac)/fac)
        elif (fac < 0):
            return PiecewisePolyExp(jumpPos=self.jumpPos/fac,
                                   left_offset=copy.deepcopy(self.right_offset),
                                   right_offset=copy.deepcopy(self.left_offset),
                                   left_polys=right_polys,
                                   right_polys=left_polys,
                                   delta_prefac=copy.deepcopy(self.delta_prefac)/fac)
        else:
            raise ValueError("fac must  not be zero")
    
    def shift(self, shift):
        r"""shifts function by given amount

        Args:
            shift(float): amount of x-shift

        Returns:
            PiecewisePolyExp: copy of function with shifted argument :math:`f(x-\text{shift})`
        """
        left_polys = [poly.shift(shift) for poly in self.get_left_polys()]
        right_polys = [poly.shift(shift) for poly in self.get_right_polys()]
        
        return PiecewisePolyExp(jumpPos=self.jumpPos+shift,
                                left_offset=copy.deepcopy(self.left_offset), 
                                right_offset=copy.deepcopy(self.right_offset),
                                left_polys=left_polys,
                                right_polys=right_polys,
                                delta_prefac=copy.deepcopy(self.delta_prefac),arg=self.x
                                )
            
    def mirror(self, pos):
        r"""mirrors function around given point

        Args:
            pos(float): mirror position

        Returns:
            PiecewisePolyExp: mirrored function :math:`f(2 \cdot \text{pos} - x)`
        """
        left_polys = [poly.mirror(pos) for poly in self.get_right_polys()]
        right_polys = [poly.mirror(pos) for poly in self.get_left_polys()]
        
        return PiecewisePolyExp(jumpPos=2*pos - self.jumpPos,
                                left_offset=copy.deepcopy(self.right_offset), 
                                right_offset=copy.deepcopy(self.left_offset),
                                left_polys=left_polys,
                                right_polys=right_polys,
                                delta_prefac=copy.deepcopy(self.delta_prefac),arg=self.x
                                )
            
    def shift_mirror(self,xl,xr):
        r"""combination of shift and mirror operation for use at periodic
        boundaries

        Args:
            xl(float): postion of mirror for the left interval
            xp(float): position of mirror for the right interval

        Returns:
            PiecewisePolyExp: mirrored and shifted function
            :math:`f(x_\text{l} + x_\text{r} - x)`
        """
        left_polys = [poly.shift_mirror(xl,xr) for poly in self.get_right_polys()]
        right_polys = [poly.shift_mirror(xl,xr) for poly in self.get_left_polys()]
        
        return PiecewisePolyExp(jumpPos=xl+xr - self.jumpPos,
                                left_offset=copy.deepcopy(self.right_offset), 
                                right_offset=copy.deepcopy(self.left_offset),
                                left_polys=left_polys,
                                right_polys=right_polys,
                                delta_prefac=copy.deepcopy(self.delta_prefac),arg=self.x
                                )
        
    def copy(self):
        r"""

        """
        left_polys = [poly.copy() for poly in self.get_left_polys()]
        right_polys = [poly.copy() for poly in self.get_right_polys()]
        
        return PiecewisePolyExp(jumpPos=self.jumpPos,
                                left_offset=copy.deepcopy(self.left_offset), 
                                right_offset=copy.deepcopy(self.right_offset),
                                left_polys=left_polys,
                                right_polys=right_polys,
                                delta_prefac=copy.deepcopy(self.delta_prefac),arg=self.x
                                )
            
        
class MultiPiecePolyExp:
    r"""representation of a sum of multiple PiecewisePolyExp objects with different
    step/delta positions.

    Supports all algebraic operations supported by PiecewisePolyExp objects.
    These will allways yield another MultipiecePolyExp object.


    Args:
        pieces(list): list of PiecewisePolyExp objects.
        arg(sympy.core.symbol.Symbol): argument of the function.
    """
    def __init__(self,pieces=[],arg=x):
        self.pieces = {}
        for piece in pieces:
            self.add_piece(piece)
        self.x = arg
        
    def add_piece(self,piece):
        r"""adds another PiecewisePolyExp object to the sum

        Args:
            piece(PiecewisePolyExp): new summand
        """
        jumpPos = piece.jumpPos
        if jumpPos in self.pieces:
            self.pieces[jumpPos] += piece
        else:
            self.pieces[jumpPos] = piece
        
            
    def as_expr(self):
        """converts MultiPiecePolyExp to a standard sympy expression"""
        res = sym.sympify(0)
        for piece in self.pieces.values():
            res += piece.as_expr()
        return res
    
    def evaluate(self, x0):
        return self.as_expr().subs(self.x,x0)
    
    def __call__(self,x0):
        return self.evaluate(x0)
    
    def __repr__(self):
        return self.as_expr().__repr__()
        
    def _repr_latex_(self):
        return self.as_expr()._repr_latex_()
    
    def __mul__(self, other):
        if self.x in sym.sympify(other).free_symbols:
            raise ValueError("can only multiply with constant factors")
        return MultiPiecePolyExp((p*other for p in self.pieces.values()))
    
    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return 1*self
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if self.x in sym.sympify(other).free_symbols:
            raise ValueError("can only divide by constant factors")
        return MultiPiecePolyExp((p/other for p in self.pieces.values()))
    
    def __add__(self, other):
        if isinstance(other, (PolyExp, PiecewisePolyExp, MultiPiecePolyExp)):
            assert self.x == other.x
        if isinstance(other, PiecewisePolyExp):
            res = self.copy()
            res.add_piece(other)
            return res
        elif isinstance(other, MultiPiecePolyExp):
            return MultiPiecePolyExp(list(self.pieces.values()) + list(other.pieces.values()),arg=self.x)
        elif isinstance(other, (int, float, sym.Expr, PolyExp)):
            if isinstance(other, sym.Expr) and self.x in other.free_symbols:
                raise ValueError("can only add constant terms")
            return MultiPiecePolyExp(pieces=[p+other if n==0 else p.copy()
                                             for n, p in enumerate(self.pieces.values())],
                                     arg=self.x)
        else:
            raise TypeError()

    def __radd__(self, other):
        return self + other
            
    def __sub__(self, other):
        if isinstance(other, (PolyExp, PiecewisePolyExp, MultiPiecePolyExp)):
            assert self.x == other.x
        if isinstance(other, PiecewisePolyExp):
            res = self.copy()
            res.add_piece(-other)
            return res
        elif isinstance(other, MultiPiecePolyExp):
            return MultiPiecePolyExp(list(self.pieces.values()) + [-p for p in other.pieces.values()])
        elif isinstance(other, (int, float, sym.Expr, PolyExp)):
            if isinstance(other, sym.Expr) and self.x in other.free_symbols:
                raise ValueError("can only subtract constant terms")
            return MultiPiecePolyExp(pieces=[p-other if n==0 else p.copy()
                                             for n, p in enumerate(self.pieces.values())],
                                     arg=self.x)
        else:
            raise TypeError()

    def __rsub__(self, other):
        return -(self-other)
            
    def exp_mul(self, exp_slope):
        return MultiPiecePolyExp((p.exp_mul(exp_slope) for p in self.pieces.values()))
    
    def exp_integrate(self, b):
        return MultiPiecePolyExp((p.exp_integrate(b) for p in self.pieces.values()))
    
    def gauss_integrate(self):
        return sum((p.gauss_integrate() for p in self.pieces.values()), sym.sympify(0))
        
    def xscale(self, fac):
        return MultiPiecePolyExp((p.xscale(fac) for p in self.pieces.values()))
    
    def shift(self, shift):
        return MultiPiecePolyExp((p.shift(shift) for p in self.pieces.values()))
    
    def mirror(self, pos):
        return MultiPiecePolyExp((p.mirror(pos) for p in self.pieces.values()))
    
    def shift_mirror(self,xl, xr):
        return MultiPiecePolyExp((p.shift_mirror(xl,xr) for p in self.pieces.values()))
    
    def copy(self):
        return MultiPiecePolyExp(pieces=[p.copy() for p in self.pieces.values()],
                                 arg=self.x)

def odd_gauss_int_helper(poly): 
    res = PolyExp(poly,-1,arg=poly.gens[0]).exp_integrate(0).poly*sym.Rational(1,2)
    assert isinstance(res, sym.Poly)
    assert res.gens == poly.gens
    return res


def odd_gauss_int(poly):
    r"""calculate the gaussian integral :math:`\int_0^x x'\mathcal{P}(x'^2) e^{-x'^2} \,\mathrm{d}x'`
    with some polynomial :math:`\mathcal{P}`.

    Args:
        poly(sympy.polys.polytools.poly): polynomial
    """
    if poly.degree() < 0:
        return 0
    arg, = poly.gens
    return odd_gauss_int_helper(poly).subs(arg,arg*arg).as_expr()*sym.exp(-arg*arg)


def even_gauss_int_helper(poly):
    if poly.degree() < 0:
        raise ValueError("zero not allowed as input")
    #print(poly)
    arg, = poly.gens
    #print(arg)
    newpoly = odd_gauss_int_helper(poly)
    #print(newpoly)
    res = arg*newpoly.subs(arg,arg**2).as_expr()
    if newpoly.degree() == 0:
        return (-arg*poly*sym.Rational(1,2),poly*sym.Rational(1,4))
    coeffs = newpoly.all_coeffs()
    nextpoly = sym.Poly([c/coeffs[0] for c in coeffs[0:-1]],arg)
    oldres, oldErrorFac = even_gauss_int_helper(nextpoly)
    #print(oldres, oldErrorFac,res)
    
    return (res -coeffs[0]*oldres, -coeffs[0]*oldErrorFac - coeffs[-1]/2)



def even_gauss_int(poly):
    r"""calculate the gaussian integral :math:`\int_0^x \mathcal{P}(x'^2) e^{-x'^2} \,\mathrm{d}x'`
    with some polynomial :math:`\mathcal{P}`.

    Args:
        poly(sympy.polys.polytools.poly): polynomial
    """
    arg, = poly.gens
    coeffs = poly.all_coeffs()
    if poly.degree() < 0:
        return 0
    if poly.degree() == 0:
        return coeffs[-1]*sym.sqrt(sym.pi)*sym.erf(arg)*sym.Rational(1,2)
    
    respoly, resErr = even_gauss_int_helper(sym.Poly(coeffs[:-1],arg))
    return respoly.as_expr()*sym.exp(-arg*arg) + (coeffs[-1]/2+resErr.as_expr())*sym.sqrt(sym.pi)*sym.erf(arg)



def time_evolution(fun,upper=sym.oo, lower=-sym.oo):
    """calculate the time evolution given the rescaled virtual initial distribution

    Because of a bug in sympy the function is at the moment not capable of handling
    infinite bounds in all but the most simple cases. As a workaround infinity should
    be approximated by an appropriately large number.

    Args:
        fun(MultiPiecePolyExp): rescaled virtual initital distribution
        upper(float): upper bound of the integration interval
        lower(float): lower bound of the integration interval

    Returns:
        sympy.core.expr.Expr: time evolution
    """
    
    x = fun.x
    antider = fun.shift(-xp).xscale(2*sym.sqrt(t)).gauss_integrate()
    if upper == sym.oo:
        int_upper = antider.subs(x,sym.oo)
    else:
        int_upper = antider.subs(x,(upper-x)/(2*sym.sqrt(t)))
        
    if lower == -sym.oo:
        int_lower = antider.subs(x,-sym.oo)
    else:
        int_lower = antider.subs(x,(lower-x)/(2*sym.sqrt(t)))
    return (int_upper - int_lower).subs(xp,x)*sym.exp(-f0*f0*t/4 + f0*x/2)/sym.sqrt(sym.pi)

# Operators to calculate virtual mirror probabilities

def mirrorRight(fl, fr, w, f0, xlp, xr, x0, fl_start):
    r"""perform the mirror operation to the right :math:`S^+` on a pair of segments of adjacent
    scaled virtual initital distributions.

    Args:
        fl(PiecewisePolyExp): segment of the left scaled virtual initial distribution
        fr(PiecewisePolyExp): segment of the right scaled virtual initial distribution
        w(float): height of the potential step
        f0(float): driving force
        xlp(float): connection point relevant for fl
        xr(float): connection point relevant for fr
        x0(float): position at which fl is known
        fl_start(float): function value of fl(x0)

    Returns:
        PiecewisePolyExp: new segment of fl
    """

    h = sym.tanh(w/2)
    
    alpha = h*f0/2
    background = PolyExp([fl_start*sym.exp(-alpha*x0)],alpha)
    
    # first integral
    if fl is not None:
        int1 = alpha*fl.mirror(xlp).exp_integrate(-alpha)
    
        int2 = h*fl.mirror(xlp).exp_mul(-alpha) + h*alpha*fl.mirror(xlp).exp_integrate(-alpha)
    else:
        int1 = 0
        int2 = 0
    
    if fr is not None:
        int3 = (1+h)*sym.exp(f0*(xr-xlp)/2)*(fr.shift(xlp-xr).exp_mul(-alpha)+alpha*fr.shift(xlp-xr).exp_integrate(-alpha))
    else:
        int3 = 0
    
    intsum = int1+int2+int3
    offset = intsum.evaluate(x0)
    intsum -= offset
    
    
    res = intsum.exp_mul(alpha)
    res += background
    return res

def mirrorLeft(fl,fr, w, f0, xlp, xr, x0, fr_start):
    r"""perform the mirror operation to the left :math:`S^+` on a pair of segments of adjacent
    scaled virtual initital distributions.

    Args:
        fl(PiecewisePolyExp): segment of the left scaled virtual initial distribution
        fr(PiecewisePolyExp): segment of the right scaled virtual initial distribution
        w(float): height of the potential step
        f0(float): driving force
        xlp(float): connection point relevant for fl
        xr(float): connection point relevant for fr
        x0(float): position at which fr is known
        fl_start(float): function value of fr(x0)

    Returns:
        PieceWisePolyExp: new segment of fr
    """

    h = sym.tanh(-w/2)
    
    alpha = h*f0/2
    background = PolyExp([fr_start*sym.exp(-alpha*x0)],alpha)
    
    if fr is not None:
        int1 = alpha*fr.mirror(xr).exp_integrate(-alpha)
        int2 = h*fr.mirror(xr).exp_mul(-alpha) + h*alpha*fr.mirror(xr).exp_integrate(-alpha)
    else:
        int1 = 0
        int2 = 0

    if fl is not None:
        int3 = (1+h)*sym.exp(f0*(xlp-xr)/2)*(fl.shift(xr-xlp).exp_mul(-alpha) + alpha*fl.shift(xr-xlp).exp_integrate(-alpha))
    else:
        int3 = 0
    
    
    intsum = int1+int2+int3
    offset = intsum.evaluate(x0)
    intsum -= offset
    
    res = intsum.exp_mul(alpha)
    res += background
    return res

# Numerical routines

def step_time_evolution_num(p0,f ,w,xmax,N, t_range):
    r"""numerically integrate the Fokker-Planck equation in case of a single potential step
    located at :math:`x=0` and free boundary conditions

    Args:
        p0(numpy.ndarray): initial distribution. Must be of length N.
        f(float): driving force
        w(float): height of the potential step
        xmax(float): defines the half-width of the discretization interval. Free boundary
                        conditions are approximated with reflective boundary conditions at :math:`x=\pm x_\text{max}`.
        N(int): number of discretized states on the x axis
        t_range(numpy.ndarray): array containing the times at which the time evolution is evaluated

    Returns:
        numpy.ndarray: Two-dimensional array containing the probability distribution at the specified times.
    """

    assert N % 2 == 0
    assert len(p0) == N
    dx = 2*xmax/(N-1)
    nl = int(N/2)-1
    nr = int(N/2)
    fp = sp.sparse.diags([1/dx**2 + f/(2*dx),-2/dx**2,1/dx**2-f/(2*dx)],[-1,0,1],shape=(N,N)).A
    fp[nr,nl] *= np.exp(-w/2)
    fp[nl,nr] *= np.exp(w/2)
    fp[nr,nr] = -(1/dx**2 + f/(2*dx) + np.exp(w/2)*(1/dx**2-f/(2*dx)))
    fp[nl,nl] = -(1/dx**2 - f/(2*dx) + np.exp(-w/2)*(1/dx**2+f/(2*dx)))
    
    return sp.integrate.odeint(lambda x,t: fp@x,p0,t_range,Dfun=lambda x,t: fp)

def double_step_evolution_num(p0, f, w1, w2, x1, x2, xmax, N, t_range):
    r"""numerically integrate the Fokker-Planck equation in case of two potential steps
    and free boundary conditions

    Args:
        p0(numpy.ndarray): initial distribution. Must be of length N.
        f(float): driving force
        w1(float): height of the left potential step
        w2(float): height of the right potential step
        x1(float): position of the left potential step
        x2(float): position of the right potential step
        xmax(float): half-width of the discretization interval.
        N(int): number of discretized states on the x-axis
        t_range(numpy.ndarray): array containing the times at which the time evolution is evaluated

    Returns:
        numpy.ndarray: Two-dimensional array containing the probability distribution at the specified times
    """

    assert N % 2 == 0
    assert len(p0) == N
    dx = 2*xmax/(N-1)
    nl1 = int((x1+xmax)/dx)
    nr1 = nl1+1
    nl2 = int((x2+xmax)/dx)
    nr2 = nl2 +1

    fp = sp.sparse.diags([1/dx**2 + f/(2*dx),-2/dx**2,1/dx**2-f/(2*dx)],[-1,0,1],shape=(N,N)).A
    fp[nr1,nl1] *= np.exp(-w1/2)
    fp[nl1,nr1] *= np.exp(w1/2)
    fp[nr1,nr1] = -(1/dx**2 + f/(2*dx) + np.exp(w1/2)*(1/dx**2-f/(2*dx)))
    fp[nl1,nl1] = -(1/dx**2 - f/(2*dx) + np.exp(-w1/2)*(1/dx**2+f/(2*dx)))
    
    fp[nr2,nl2] *= np.exp(-w2/2)
    fp[nl2,nr2] *= np.exp(w2/2)
    fp[nr2,nr2] = -(1/dx**2 + f/(2*dx) + np.exp(w2/2)*(1/dx**2-f/(2*dx)))
    fp[nl2,nl2] = -(1/dx**2 - f/(2*dx) + np.exp(-w2/2)*(1/dx**2+f/(2*dx)))
    
    return sp.integrate.odeint(lambda x,t: fp@x, p0, t_range, Dfun=lambda x,t: fp)

def multi_step_evolution_num(p0, f, w_list, x_list, xmax, N, t_range):
    r"""numerically integrate the Fokker-Planck equation in case of an arbitrary number of potential
    steps and free boundary conditions

    Args:
        p0(numpy.ndarray): initial distribution. Must be of length N.
        f(float): driving force
        w_list(list): list of heights of the potential steps
        x_list(list): list of the positions of the potential steps
        xmax(float): Half-width of the discretization interval
        N(int): number of discretized states on the x-axis
        t_range(numpy.ndarray): array containing the times at which the time evolution is evaluated.

    Returns:
        numpy.ndarray: Two-dimensional array containing the probability distribution at the specified times.
    """

    assert N % 2 == 0
    assert len(p0) == N
    dx = 2*xmax/(N-1)
    
    fp = sp.sparse.diags([1/dx**2 + f/(2*dx),-2/dx**2,1/dx**2-f/(2*dx)],[-1,0,1],shape=(N,N)).A
    
    for wi, xi in zip(w_list, x_list):
    
        nl = int((xi+xmax)/dx)
        nr = nl+1

        fp[nr,nl] *= np.exp(-wi/2)
        fp[nl,nr] *= np.exp(wi/2)
        fp[nr,nr] = -(1/dx**2 + f/(2*dx) + np.exp(wi/2)*(1/dx**2-f/(2*dx)))
        fp[nl,nl] = -(1/dx**2 - f/(2*dx) + np.exp(-wi/2)*(1/dx**2+f/(2*dx)))
    
    return sp.integrate.odeint(lambda x,t: fp@x, p0, t_range, Dfun=lambda x,t: fp)

def periodic_step_evolution_num(p0, f, w, xp, N, t_range):
    r"""numerically integrate the Fokker-Planck equation in case of a single single potential step
    and periodic boundary conditions

    Args:
        p0(numpy.ndarray): initital distribution. Must be of length N.
        f(float): driving force
        w(float): height of the potential step
        xp(float): period length
        N(int): number of discretized states on the x-axis
        t_range(numpy.ndarray): array containing the the times at which the time evolution is evaluated.

    Returns:
        numpy.ndarray: two-dimensional array containing the probability distribution at the specified times
    """

    assert len(p0) == N
    dx = xp/(N-1)
    
    fp = sp.sparse.diags([1/dx**2 + f/(2*dx),-2/dx**2,1/dx**2-f/(2*dx)],[-1,0,1],shape=(N,N)).A
    fp[-1,0] = np.exp(w/2)*(1/dx**2-f/(2*dx))
    fp[0,-1] = np.exp(-w/2)*(1/dx**2 + f/(2*dx))
    fp[0,0] =  -(1/dx**2 + f/(2*dx) + np.exp(w/2)*(1/dx**2-f/(2*dx)))
    fp[-1,-1] = -(1/dx**2 - f/(2*dx) + np.exp(-w/2)*(1/dx**2+f/(2*dx)))
    
    return sp.integrate.odeint(lambda x,t: fp@x, p0, t_range, Dfun=lambda x,t: fp)

def periodic_multi_step_evolution_num(p0, f, w_list, x_list, N, t_range):
    r"""numerically integrate the Fokker-Planck equation in case of multiple potential steps
    and periodic boundary conditions.

    Args:
        p0(numpy.ndarray): initial distribution. Must be of length N.
        f(float): driving force
        w_list(list): list of heights of the potential steps
        x_list(list): list of the positions of the potential steps. The last step is assumed
            to be located at the periodic boundary, so x_list[-1] is also the period length.
        N(int): number of discretized states on the x-axis
        t_range(numpy.ndarray): array containing the times at which the time evolution is evaluated.

    Returns:
        numpy.ndarray: two-dimensional array containing the probability distribution at the specified times
    """

    assert len(p0) == N
    assert np.diff(x_list) >= 0
    assert len(w_list) == len(x_list)
    dx = x_list[-1]/(N-1)
    
    fp = sp.sparse.diags([1/dx**2 + f/(2*dx),-2/dx**2,1/dx**2-f/(2*dx)],[-1,0,1],shape=(N,N)).A
    for wi, xi in zip(w_list[:-1],x_list[:-1]):
        nl = int(xi/dx)
        nr = nl+1

        fp[nr,nl] *= np.exp(-wi/2)
        fp[nl,nr] *= np.exp(wi/2)
        fp[nr,nr] = -(1/dx**2 + f/(2*dx) + np.exp(wi/2)*(1/dx**2-f/(2*dx)))
        fp[nl,nl] = -(1/dx**2 - f/(2*dx) + np.exp(-wi/2)*(1/dx**2+f/(2*dx)))
        
    w = w_list[-1]

    fp[-1,0] = np.exp(w/2)*(1/dx**2-f/(2*dx))
    fp[0,-1] = np.exp(-w/2)*(1/dx**2 + f/(2*dx))
    fp[0,0] =  -(1/dx**2 + f/(2*dx) + np.exp(w/2)*(1/dx**2-f/(2*dx)))
    fp[-1,-1] = -(1/dx**2 - f/(2*dx) + np.exp(-w/2)*(1/dx**2+f/(2*dx)))
    
    return sp.integrate.odeint(lambda x,t: fp@x, p0, t_range, Dfun=lambda x,t: fp)

def flat_time_evolution_num(p0, f, xmax, N, t_range):
    r"""numerically integrate the Fokker-Planck equation in absence of a potential step

    Args:
        p0(numpy.ndarray): initial distribution. Must be of length N.
        f(float): driving force
        xmax(float): half-width of the discretization interval
        N(int): number of discretized states on the x-axis.
        t_range(numpy.ndarray): array containing the times at which the times evolution is evaluated.

    Returns:
        numpy.ndarray: two-dimensional array containing the probability distribution at the specified times.
    """

    assert len(p0) == N
    dx = 2*xmax/(N-1)
    fp = sp.sparse.diags([1/dx**2 + f/dx,-2/dx**2,1/dx**2-f/dx],[-1,0,1],shape=(N,N)).A
    
    return sp.integrate.odeint(lambda x,t: fp@x,p0,t_range,Dfun=lambda x,t: fp)

# Finished routines for example systems
def gen_numerical_time_evolution(func, f0_num):
    r"""generates a python function that implements the analytic approximation of some time evolution

    Args:
        func(sympy.core.expr.Expr): approximation to the time evolution obtained by calling :meth:`StepPotential.time_evolution`
        f0_num(float): numerical value of the driving force

    Returns:
        FunctionType: numerical respresentation of the time evolution
    """

    te_subs = func.subs(f0,f0_num)
    return sym.lambdify([x,t],te_subs,
                        modules=[{'Heaviside':lambda x: np.heaviside(x,0.5) },'numpy', 'scipy'])

def double_step_evolution_approx(f0, w1, w2, x1, x2, N, x0,simplify=False,numerical_result=False,verbose=False, extended_range=None):
    r"""calculate analytic approximation to the propagator in case of a potential with two
    steps with free boundary conditions.
    
    Args:
        f0(float): driving force
        w1(float): height of the left potential step
        w2(float): height of the right potential step
        x1(float): position of the left potential step
        x2(float): position of the right potential step
        N(int): number of mirror iterations
        x0(float): initial position
        simplify(bool): apply sympy simplification to the result (slow)
        numerical_result(bool): return a numerical implementation of the function rather than a 
            analytical expression
        verbose(bool): turn verbose debugging output on
        extended_range(tuple): expects tuple of the form (lower, upper). When provided the integration
            resulting in the time evolution is provided in the interval (lower, upper) rather than 
            the interval for which the virtual initial distribution is known.
        

    Returns:
        sympy.core.expr.Expr: propagator
    """

    width = x2-x1
    startsec = int(np.floor((x0-x1)/width))

    offset = x0 - (x1+startsec*width) if startsec %2 == 0 else (x1 +(startsec+1)*width) - x0
    init_prefac = sym.exp(-f0*x0*sym.Rational(1,2))

    def get_jump_pos(section):
        #print(section, width, offset, x1)
        if section %2 == 0:
            return x1+section*width+offset
        else:
            return x1 + (section+1)*width - offset
        
    p1_pieces = deque([(x1-width, x1, PiecewisePolyExp(jumpPos=get_jump_pos(-1),
                                    delta_prefac=init_prefac if x1-width < x0 < x1 else 0))]
                     )
    p2_pieces = deque([(x1,x1+width, PiecewisePolyExp(jumpPos=get_jump_pos(0),
                                   delta_prefac=init_prefac if x1 < x0 <x1+width else 0))]
                     )
    p3_pieces = deque([(x1+width,x1+2*width, PiecewisePolyExp(jumpPos=get_jump_pos(1),
                                   delta_prefac=init_prefac if x1+width < x0 < x1+2*width else 0))]
                     )
    
    for i in range(N):

        new_p1_left = PiecewisePolyExp(jumpPos=get_jump_pos(-2-i),delta_prefac=0)
        
        new_p1_right = mirrorRight(p1_pieces[0][2],p2_pieces[-1][2],w1,f0,x1,x1,
                                   x0=x1+i*width,fl_start=p1_pieces[-1][2](x1+i*width))

        new_p2_left = mirrorLeft(p1_pieces[0][2],p2_pieces[-1][2],w1,f0,x1,x1,
                                x0=x1-i*width,fr_start=p2_pieces[0][2](x1-i*width))
 
        new_p2_right = mirrorRight(p2_pieces[0][2],p3_pieces[-1][2],w2,f0,x2,x2,
                                  x0=x2+i*width,fl_start=p2_pieces[-1][2](x2+i*width))

        new_p3_left = mirrorLeft(p2_pieces[0][2],p3_pieces[-1][2],w2,f0,x2,x2,
                                x0=x2-i*width, fr_start=p3_pieces[0][2](x2-i*width))

        new_p3_right = PiecewisePolyExp(jumpPos=get_jump_pos(2+i),delta_prefac=0)
              
        if verbose:
            print(i)
        
            print("combining p1 p2 right")
            print(p1_pieces[0],p2_pieces[-1])
            print("seam")
            print(x1+i*width)
            print("result")
            print(new_p1_right)
            print()
            
            print("combining p1 p2 left")
            print(p1_pieces[0],p2_pieces[-1])
            print("seam")
            print(x1-i*width)
            print("result")
            print(new_p2_left)
            print()
            
            print("combining p2 p3 right")
            print(p2_pieces[0],p3_pieces[-1])
            print("seam")
            print(x2+i*width)
            print(p2_pieces[-1])
            print("result")
            print(new_p2_right)
            print()

            print("combining p2 p3 left")
            print(p2_pieces[0],p3_pieces[-1])
            print("seam")
            print(x2-i*width)
            print("result")
            print(new_p3_left)
            print()
            print()
        
        p1_pieces.appendleft((x1-(2+i)*width, x1-(1+i)*width, new_p1_left))
        p1_pieces.append((x1+i*width, x1+(i+1)*width, new_p1_right))
        p2_pieces.appendleft((x1-(i+1)*width, x1-i*width, new_p2_left))
        p2_pieces.append((x1+(i+1)*width,x1+(i+2)*width, new_p2_right))
        p3_pieces.appendleft((x1-i*width, x1-(i-1)*width, new_p3_left))
        p3_pieces.append((x1+(i+2)*width, x1+(i+3)*width, new_p3_right))

    p1_te = 0
    for n,(x_lower, x_upper, func) in enumerate(p1_pieces):
        if extended_range is not None and n==len(p1_pieces)-1:
            new_te = time_evolution(func,extended_range[1], x_lower)
        else:
            new_te = time_evolution(func,x_upper, x_lower)
  
        if simplify:
            p1_te += new_te.simplify()
        else:
            p1_te += new_te
            
    p2_te = 0
    for n,(x_lower, x_upper, func) in enumerate(p2_pieces):
        if extended_range is not None and n==0:
            new_te = time_evolution(func,x_upper, extended_range[0])
        elif extended_range is not None and n==len(p1_pieces)-1:
            new_te = time_evolution(func,extended_range[1], x_lower)       
        else:
            new_te = time_evolution(func,x_upper, x_lower)
            
        if simplify:
            p2_te += new_te.simplify()
        else:
            p2_te += new_te
            
    p3_te = 0
    for n,(x_lower, x_upper, func) in enumerate(p3_pieces):
        if extended_range is not None and n==0:
            new_te = time_evolution(func,x_upper, extended_range[0])
        else:
            new_te = time_evolution(func, x_upper, x_lower)
            
        if simplify:
            p3_te += new_te.simplify()
        else:
            p3_te += new_te
            
    if numerical_result:
        return (gen_numerical_time_evolution(p1_te, f0),
               gen_numerical_time_evolution(p2_te, f0),
               gen_numerical_time_evolution(p3_te, f0))
    
    return p1_te, p2_te, p3_te


def multi_step_evolution_approx(f0, w, x1, width, N, x0, simplify=False,numerical_result=False,num_prefactor=False, extended_range=None):
    r"""calculate analytic approximation to the propagator in case of an arbitrary number of equidistant potential steps with free boundary conditions.

    Args:
        f0(float): driving force
        w(list): list of heights of the potential steps
        x1(float): position of the first potential step
        width(float): distance of neighboring potential steps
        N(int): number of mirror iterations
        x0(float): initial position
        simplify(bool): apply sympy simplification to the result (slow)
        numerical_result(bool): return a numerical implementation of the function rather than a 
            analytical expression
        verbose(bool): turn verbose debugging output on
        extended_range(tuple): expects tuple of the form (lower, upper). When provided the integration
            resulting in the time evolution is provided in the interval (lower, upper) rather than 
            the interval for which the virtual initial distribution is known.
        
    Returns:
        sympy.core.expr.Expr: propagator
    """

    num_sec = len(w)+1
    startsec = int(np.floor((x0-x1)/width))
    
    offset = x0 - (x1+startsec*width) if startsec %2 == 0 else (x1 +(startsec+1)*width) - x0
    if num_prefactor:
        init_prefac = np.exp(-f0*x0/2)
    else:
        init_prefac = sym.exp(-f0*x0*sym.Rational(1,2))
    
    def get_jump_pos(section):
        
        if section %2 == 0:
            return x1+section*width+offset
        else:
            return x1 + (section+1)*width - offset
        
    pn_pieces = []
    for n in range(num_sec):
        pn_pieces.append(deque([
            (x1+(n-1)*width,x1+n*width, PiecewisePolyExp(jumpPos=get_jump_pos(n-1),
                                    delta_prefac=init_prefac if x1+(n-1)*width < x0 < x1+n*width else 0))
        ]))
        
    for step in range(N):
        new_left = []
        new_right = []
        for n in range(num_sec):
            if n == 0:
                new_left.append(PiecewisePolyExp(jumpPos=get_jump_pos(-2-step), delta_prefac=0))
            else:
                connection_pos = x1+(n-step-1)*width
                new_left.append(mirrorLeft(pn_pieces[n-1][0][2], pn_pieces[n][-1][2],w[n-1], f0,
                                          x1+(n-1)*width, x1+(n-1)*width,
                                           x0=connection_pos, fr_start=pn_pieces[n][0][2](connection_pos))
                               )
            if n == num_sec-1:
                new_right.append(PiecewisePolyExp(jumpPos=get_jump_pos(-1+num_sec+step), delta_prefac=0))
            else:
                connection_pos = x1+(n+step)*width
                new_right.append(mirrorRight(pn_pieces[n][0][2],pn_pieces[n+1][-1][2],w[n], f0,
                                            x1+n*width, x1+n*width,
                                            x0=connection_pos, fl_start=pn_pieces[n][-1][2](connection_pos))
                                )
        for n in range(num_sec):
            pn_pieces[n].appendleft((x1-(2-n+step)*width, x1-(1-n+step)*width, new_left[n]))
            pn_pieces[n].append((x1+(n+step)*width, x1+(n+step+1)*width, new_right[n]))
            
    p_te = [0 for _ in range(num_sec)]
    for n in range(num_sec):
        for i, (x_lower, x_upper, func) in enumerate(pn_pieces[n]):
            if extended_range is not None and i == 0:
                new_te = time_evolution(func, x_upper, extended_range[0])
            elif extended_range is not None and i == len(pn_pieces[n]) -1:
                new_te = time_evolution(func, extended_range[1], x_lower)
            else:
                new_te = time_evolution(func, x_upper, x_lower)

            if simplify:
                p_te[n] += new_te.simplify()
            else:
                p_te[n] += new_te
            
    if numerical_result:
        return tuple(gen_numerical_time_evolution(pi, f0) for pi in p_te)
    else:
        return p_te


def periodic_step_evolution_approx(f0, w, xp, N, x0, simplify=False,
                                   numerical_result=False, num_prefac=False, extended_range=None):
    r"""calculate analytic approximation to the propagator in case of a single potential step
    and periodic boundary conditions.

    Args:
        f0(float): driving force
        w(float): height of the potential step
        xp(float): period length
        N(int): number of mirror iterations
        x0(float): initial position
        simplify(bool): apply sympy simplification to the result (slow)
        numerical_result(bool): return a numerical implementation of the function rather than a 
            analytical expression
        verbose(bool): turn verbose debugging output on
        extended_range(tuple): expects tuple of the form (lower, upper). When provided the integration
            resulting in the time evolution is provided in the interval (lower, upper) rather than 
            the interval for which the virtual initial distribution is known.
        
    Returns:
        sympy.core.expr.Expr: propagator

    """

    assert x0 < xp
    if num_prefac:
        init_prefac = np.exp(-f0*x0/2)
    else:
        init_prefac = sym.exp(-f0*x0*sym.Rational(1,2))
        
    pieces = deque([(0,xp,PiecewisePolyExp(x0,delta_prefac=init_prefac))])
    
    for step in range(N):
        connection_pos = pieces[-1][1]
        new_right = mirrorRight(pieces[0][2], pieces[-1][2],w,f0,xp, 0,
                                connection_pos, pieces[-1][2](connection_pos))
        
        connection_pos = pieces[0][0]
        new_left = mirrorLeft(pieces[0][2], pieces[-1][2], w, f0, xp, 0,
                              connection_pos, pieces[0][2](connection_pos))
        
        pieces.appendleft((-(step+1)*xp, -step*xp, new_left))
        pieces.append(((step+1)*xp, (step+2)*xp, new_right))
        
        
    p_te = 0
    for i, (x_lower, x_upper, func) in enumerate(pieces):
        if extended_range is not None and i == 0:
            new_te = time_evolution(func, x_upper, extended_range[0])
        elif extended_range is not None and i == len(pieces) -1:
            new_te = time_evolution(func, extended_range[1], x_lower)
        else:
            new_te = time_evolution(func, x_upper, x_lower)
            
        if simplify:
            p_te += new_te.simplify()
        else:
            p_te += new_te
            
    
    if numerical_result:
        return gen_numerical_time_evolution(p_te, f0)
    else:
        return p_te


def periodic_multi_step_evolution_approx(f0, w, x1, width, N, x0, simplify=False,numerical_result=False,num_prefactor=False, extended_range=None):
    r"""calculate analytic approximation to the propagator in case of an arbitrary number of
    equidistant potential steps with periodic boundary conditions.

    Args:
        f0(float): driving force
        w(list): list of heights of the potential steps
        x1(float): position of the first potential step
        width(float): distance of neighboring potential steps
        N(int): number of mirror iterations
        x0(float): initial position
        simplify(bool): apply sympy simplification to the result (slow)
        numerical_result(bool): return a numerical implementation of the function rather than a 
            analytical expression
        verbose(bool): turn verbose debugging output on
        extended_range(tuple): expects tuple of the form (lower, upper). When provided the integration
            resulting in the time evolution is provided in the interval (lower, upper) rather than 
            the interval for which the virtual initial distribution is known.
        
    Returns:
        sympy.core.expr.Expr: propagator
    """

    num_sec = len(w)
    if num_prefactor:
        init_prefac = np.exp(-f0*x0/2)
    else:
        init_prefac = sym.exp(-f0*x0*sym.Rational(1,2))
    
        
    pn_pieces = []
    for n in range(num_sec):
        pn_pieces.append(deque([
            (x1+n*width,x1+(n+1)*width, PiecewisePolyExp(jumpPos=x0,delta_prefac=init_prefac) if x1+n*width < x0 < x1+(n+1)*width else MultiPiecePolyExp())
        ]))
        
    for step in range(N):
        new_left = []
        new_right = []
        for n in range(num_sec):
            if n == 0:
                connection_pos = pn_pieces[0][0][0]
                new_left.append(mirrorLeft(pn_pieces[-1][0][2], pn_pieces[0][-1][2],w[-1], f0,
                                           x1+num_sec*width, x1,
                                           x0=connection_pos, fr_start=pn_pieces[0][0][2](connection_pos))
                               )
            else:
                connection_pos = x1+(n-step)*width
                new_left.append(mirrorLeft(pn_pieces[n-1][0][2], pn_pieces[n][-1][2],w[n-1], f0,
                                          x1+n*width, x1+n*width,
                                           x0=connection_pos, fr_start=pn_pieces[n][0][2](connection_pos))
                               )
            if n == num_sec-1:
                connection_pos = pn_pieces[-1][-1][1]
                new_right.append(mirrorRight(pn_pieces[-1][0][2], pn_pieces[0][-1][2],w[-1], f0,
                                            x1+num_sec*width, x1,
                                            x0=connection_pos, fl_start=pn_pieces[-1][-1][2](connection_pos)))
            else:
                connection_pos = x1+(n+step+1)*width
                new_right.append(mirrorRight(pn_pieces[n][0][2],pn_pieces[n+1][-1][2],w[n], f0,
                                            x1+(n+1)*width, x1+(n+1)*width,
                                            x0=connection_pos, fl_start=pn_pieces[n][-1][2](connection_pos))
                                )
        for n in range(num_sec):
            pn_pieces[n].appendleft((x1-(1-n+step)*width, x1-(-n+step)*width, new_left[n]))
            pn_pieces[n].append((x1+(n+1+step)*width, x1+(n+step+2)*width, new_right[n]))
            
            
    p_te = [0 for _ in range(num_sec)]
    for n in range(num_sec):
        for i, (x_lower, x_upper, func) in enumerate(pn_pieces[n]):
            if extended_range is not None and i == 0:
                new_te = time_evolution(func, x_upper, extended_range[0])
            elif extended_range is not None and i == len(pn_pieces[n]) -1:
                new_te = time_evolution(func, extended_range[1], x_lower)
            else:
                new_te = time_evolution(func, x_upper, x_lower)

            if simplify:
                p_te[n] += new_te.simplify()
            else:
                p_te[n] += new_te
            
    if numerical_result:
        return tuple(gen_numerical_time_evolution(pi, f0) for pi in p_te)
    else:
        return p_te

