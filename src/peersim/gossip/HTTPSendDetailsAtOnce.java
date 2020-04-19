package peersim.gossip;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.Socket;
import java.net.URL;
import java.net.URLEncoder;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

public class HTTPSendDetailsAtOnce {
	
	public static void sendRequest(String base_url, String command, JsonObject nnconfig)
	{
	
		HttpURLConnection connection = null;  

		//Then credentials and send string
	    String send_string = base_url + "/" + command;
	    
	    ///First, all the GSON/JSon stuff up front
        Gson gson = new Gson();
        //convert java object to JSON format
        String json = gson.toJson(nnconfig);
	    
        try {
	      //Create connection
	      URL url = new URL("http://127.0.0.1:5000/"+send_string);
	      String urlParameters = "nnconfig=" + URLEncoder.encode(json, "UTF-8");
	      connection = (HttpURLConnection)url.openConnection();
	      connection.setRequestMethod("POST");
	      connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
	
	      connection.setRequestProperty("Content-Length", "" + Integer.toString(urlParameters.getBytes().length));
	      connection.setRequestProperty("Content-Language", "en-US");  
	
	      connection.setUseCaches(false);
	      connection.setDoInput(true);
	      connection.setDoOutput(true);
	      
	
	      //Send request
	      DataOutputStream wr = new DataOutputStream (connection.getOutputStream ());
	      wr.writeBytes (urlParameters);
	      wr.flush ();
	      wr.close ();
	
	      //Get Response    
	      InputStream is = connection.getInputStream();
	      
	      BufferedReader rd = new BufferedReader(new InputStreamReader(is));
	      String line;
	      StringBuffer response = new StringBuffer(); 
	      while((line = rd.readLine()) != null) {
	        response.append(line);
	        response.append('\r');
	      }
	      rd.close();
	      
	      System.out.println(response);
	      
	    } catch (Exception e) {
	        e.printStackTrace();
	        //return null;
	    }finally {
	        if(connection != null) {
	            connection.disconnect(); 
	        }
	    }
	}
	
	
	public static void main(String args[]) {
		
	}

}
