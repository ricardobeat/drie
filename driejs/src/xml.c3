// XML for js/dom.js's DOMParser, parsed by the stdlib's std::encoding::xml (yxml).
//
// xmlParse(text) returns the document's events as one string of pairs of fields, each
// field ended by a NUL (which XML 1.0 does not allow in a document, so it never appears in
// one). The first field of a pair is a kind and a name, the second a value:
//
//   "<" name, ""       an element starts (its attributes follow)
//   "=" name, value    an attribute of the element that just started
//   ">", ""            an element ends
//   "#", text          text, CDATA included, with entities and character references decoded
//   "?" target, data   a processing instruction
//   "!" error, ""      the parse failed here: the last pair
//
// One string and one split in JS cost far less than a JS value per node, for a file of
// thousands of elements (a Collada model).
module main;

import boomkat;
import std::io, std::encoding::xml;

// yxml's stack holds the names of the open elements and of the current attribute
const sz XML_STACK = 64 * 1024;

fn void h_xml_parse(JsCtx ctx, void* udata)
{
    @pool()
    {
        String? text = ctx.arg(0).as_string();
        if (catch text)
        {
            ctx.throw_error(TYPE, "xmlParse: expected a string");
            return;
        }
        DString out;
        out.init(tmem, text.len + text.len / 4 + 64);
        ByteReader bytes;
        bytes.init(text);
        XmlReader reader;
        char[] stack = mem::temp_array(char, XML_STACK);
        if (catch err = reader.init(&bytes, stack, tmem))
        {
            xml_error(&out, err, &reader);
            ctx.ret_string(out.str_view());
            return;
        }
        defer reader.destroy();
        while (true)
        {
            XmlEvent? event = reader.next_event();
            if (catch err = event)
            {
                if (err != io::EOF) xml_error(&out, err, &reader);
                break;
            }
            switch (event.kind)
            {
                case START_ELEMENT: xml_pair(&out, '<', event.name, "");
                case ATTRIBUTE: xml_pair(&out, '=', event.name, event.value);
                case END_ELEMENT: xml_pair(&out, '>', "", "");
                case TEXT: xml_pair(&out, '#', "", event.value);
                case PI: xml_pair(&out, '?', event.name, event.value);
                case COMMENT: break;
            }
        }
        ctx.ret_string(out.str_view());
    };
}

fn void xml_pair(DString* out, char kind, String name, String value) @inline
{
    out.append_char(kind);
    out.append(name);
    out.append_char('\0');
    out.append(value);
    out.append_char('\0');
}

fn void xml_error(DString* out, fault err, XmlReader* reader)
{
    String why;
    switch (err)
    {
        case xml::XML_SYNTAX: why = "syntax error";
        case xml::XML_TAG_MISMATCH: why = "mismatched tag";
        case xml::XML_INVALID_REF: why = "invalid entity or character reference";
        case xml::XML_STACK_OVERFLOW: why = "elements nested too deeply";
        case io::UNEXPECTED_EOF: why = "unexpected end of document";
        default: why = "cannot be parsed";
    }
    xml_pair(out, '!', string::tformat("XML Parsing Error: %s at line %d, column %d", why,
        reader.parser.line, reader.parser.byte), "");
}
