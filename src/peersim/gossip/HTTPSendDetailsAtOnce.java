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
	
	public static void sendCommand(String command, int node_id)
	{
	
		HttpURLConnection connection = null;  

	  //Then credentials and send string
	    String send_string = command+"_"+node_id;
	
	    try {
	      //Create connection
	      URL url = new URL("http://127.0.0.1:5000/update_project/"+send_string);
	      String urlParameters = "command=" + command+"+"+"node_id="+node_id;
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
		HTTPSendDetailsAtOnce newConnection = new HTTPSendDetailsAtOnce();
		newConnection.sendCommand("hello", 1);
		newConnection.sendCommand("hello", 2);
		newConnection.sendCommand("hello", 3);
		
	}

}