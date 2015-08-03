package com.github.mrsdogood.neural;

public class ActNaNException extends RuntimeException{
    public ActNaNException(String m){
        super(m);
    }
    public ActNaNException(){
        super();
    }
}
