package com.example.ai.tensorflow.controller;

import com.example.ai.tensorflow.service.MultiplyService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;

@Controller
@RequestMapping("/api/")
public class MultiplyController {

    @Autowired
    MultiplyService service;

    @ResponseBody
    @RequestMapping(value = "/double", method = RequestMethod.POST)
    @ApiOperation(value = "Double a number.",notes = "Double a number")
    public String doWork(@RequestParam(name = "number", defaultValue = "15") float number) {
        return number + " * 2.0 = " + service.get(new float[]{number})[0];
    }
}
